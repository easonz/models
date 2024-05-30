from six.moves import map, zip
from functools import partial
import numpy as np


import mindspore
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.dataset import vision
from mindspore.dataset.vision.utils import Inter

from .conv_model import Conv2D_Compatible_With_Torch
from . import mmcv_utils
from .focal_loss_ms import SigmoidFoaclLoss
import logging

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def points_nms(heat, kernel=2):
    # kernel must be 2
    ori_type = heat.dtype
    heat = heat.astype(ms.float16)
    hmax = ms.ops.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    #z3005003 max_pool2d
    # hmax = hmax.astype(ori_type)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = ms.ops.mm(seg_masks, seg_masks.swapaxes(1, 0))
    # union.
    shape = (n_samples, n_samples)
    sum_masks_x = sum_masks.broadcast_to(shape)
    # iou.
    iou_matrix = ms.ops.triu((inter_matrix / (sum_masks_x + sum_masks_x.swapaxes(1, 0) - inter_matrix)), diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.broadcast_to(shape)
    label_matrix = ms.ops.triu((cate_labels_x == cate_labels_x.swapaxes(1, 0)).float(), diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0, return_indices=True)
    shape = (n_samples, n_samples)
    compensate_iou = compensate_iou.broadcast_to(shape).swapaxes(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = ms.ops.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = ms.ops.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0, return_indices=True)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0, return_indices=True)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update

def dice_loss(input, target):
    #z30055003
    # input = input.view(input.shape[0], -1)
    # target = target.view(target.shape[0], -1).float()
    input = input.contiguous().view(input.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1).float()
 
    a = mindspore.ops.sum(input * target, 1)
    b = mindspore.ops.sum(input * input, 1) + 0.001
    c = mindspore.ops.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d
 
def center_of_mass(bitmasks):
    _, h, w = bitmasks.shape
    # logging.info(f'bitmask type:{type(bitmasks)} size:{_}, {h}, {w}')
    ys = ms.ops.arange(0, h, dtype=ms.float32)
    xs = ms.ops.arange(0, w, dtype=ms.float32)
    m00 = bitmasks.sum(axis=-1,dtype=ms.int64).sum(axis=-1,dtype=ms.int64).clamp(min=1e-6)
    # import pdb;pdb.set_trace()
    m10 = (bitmasks * xs).sum(axis=-1).sum(axis=-1)
    m01 = (bitmasks * ys[:, None]).sum(axis=-1).sum(axis=-1)
    # logging.info(f'm00:{m00}\n m10:{m10}\n m01:{m01}')
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

class SOLOv2Head_ms(ms.nn.Cell):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None):
        super().__init__(auto_prefix=True)
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges

        self.loss_cate = self.build_focal_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = ms.nn.CellList()
        self.kernel_convs = ms.nn.CellList()
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                Conv2D_Compatible_With_Torch(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                Conv2D_Compatible_With_Torch(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.solo_cate = Conv2D_Compatible_With_Torch(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1, activation=None).to_float(ms.float16)

        self.solo_kernel = Conv2D_Compatible_With_Torch(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1, activation=None).to_float(ms.float16)

        
        
        
    def construct(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.shape[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        cate_pred, kernel_pred = multi_apply(self.construct_single, list(new_feats),
                                             list(range(len(self.seg_num_grids))),
                                             eval=eval, upsampled_size=upsampled_size)
        return cate_pred, kernel_pred

    
    def split_feats(self, feats):
        return (F.interpolate(feats[0].astype(ms.float32), size=(int(feats[0].shape[-2] * 0.5), int(feats[0].shape[-1] * 0.5)), mode='bilinear'),
        feats[1].astype(ms.float32),
        feats[2].astype(ms.float32),
        feats[3].astype(ms.float32),
        F.interpolate(feats[4].astype(ms.float32), size=feats[3].shape[-2:], mode='bilinear'))

    def construct_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = ms.numpy.linspace(-1, 1, ins_kernel_feat.shape[-1])
        y_range = ms.numpy.linspace(-1, 1, ins_kernel_feat.shape[-2])
        y, x = ms.numpy.meshgrid(y_range, x_range)

        x = x.T
        y = y.T

        y = F.broadcast_to(y, tuple([ins_kernel_feat.shape[0], 1, -1, -1]))
        x = F.broadcast_to(x, tuple([ins_kernel_feat.shape[0], 1, -1, -1]))

        coord_feat = ms.ops.cat([x, y], axis=1)
        # ori_type0 = ins_kernel_feat.dtype
        # ori_type1 = coord_feat.dtype
        # ins_kernel_feat = ins_kernel_feat.astype(ms.float16)
        # coord_feat = coord_feat.astype(ms.float16)
        # print('ins_kernel_feat',ins_kernel_feat.dtype, coord_feat.dtype)
        ins_kernel_feat = ms.ops.cat([ins_kernel_feat.astype(ms.float32), coord_feat], axis=1)
        # ins_kernel_feat = ins_kernel_feat.astype(ori_type0)
        # coord_feat = coord_feat.astype(ori_type1)

        # original
        # coord_feat = ms.ops.functional.concat([x, y], axis=1)
        # ori_type0 = ins_kernel_feat.dtype
        # ori_type1 = coord_feat.dtype
        # ins_kernel_feat = ins_kernel_feat.astype(ms.float16)
        # coord_feat = coord_feat.astype(ms.float16)
        # ins_kernel_feat = ms.ops.functional.concat([ins_kernel_feat, coord_feat], axis=1)
        # ins_kernel_feat = ins_kernel_feat.astype(ori_type0)
        # coord_feat = coord_feat.astype(ori_type1)

        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]

        ori_type = kernel_feat.dtype
        # import pdb;pdb.set_trace()
        kernel_feat = kernel_feat.astype(ms.float32)
        #156 204 269 322 resize
        # print("***************************333 : ",kernel_feat.dtype)
        kernel_feat = F.interpolate(kernel_feat, size=(seg_num_grid, seg_num_grid), mode='bilinear')
        kernel_feat = kernel_feat.astype(ori_type)

        cate_feat = kernel_feat[:, :-2, :, :]

        # kernel_feat = kernel_feat.contiguous()

        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)

        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        # cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred
    

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        # shape = seg_pred.shape
        # size = [(shape[0], shape[1], shape[2], shape[3])]
        featmap_size = seg_pred.shape[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].reshape(-1, self.cate_out_channels) for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).reshape(-1, self.kernel_out_channels)
                                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            cate_pred_list = ms.ops.cat(cate_pred_list, axis=0)
            kernel_pred_list = ms.ops.cat(kernel_pred_list, axis=0)

            # original
            # cate_pred_list = ms.ops.functional.concat(cate_pred_list, axis=0)
            # kernel_pred_list = ms.ops.functional.concat(kernel_pred_list, axis=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                    featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list
    
    def get_seg_single(self, 
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        assert len(cate_preds) == len(kernel_preds)
        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # print("inds",inds)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = ms.tensor(self.seg_num_grids, dtype=cate_labels.dtype).pow(2)
        ori_type = size_trans.dtype
        size_trans = size_trans.astype(ms.float16).cumsum(0)
        size_trans = size_trans.astype(ori_type)
        strides = ms.numpy.ones((int(size_trans[-1].numpy()),))

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]
        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        ori_type0 = seg_preds.dtype
        ori_type1 = kernel_preds.dtype
        seg_preds = seg_preds.astype(ms.float16)
        kernel_preds = kernel_preds.astype(ms.float16)
        seg_preds = ms.ops.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        seg_preds = seg_preds.astype(ori_type0)
        kernel_preds = kernel_preds.astype(ori_type1)
        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()
        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None
        input_indices  = []
        temp = 0

        # keep = keep.asnumpy().tolist()

        # seg_masks = seg_masks[keep, ...] # bool
        # seg_masks = seg_masks.asnumpy()
        seg_masks = seg_masks[keep, ...]
        # seg_masks = ms.Tensor(seg_masks)

        # seg_preds = seg_preds.asnumpy()
        seg_preds = seg_preds[keep, ...] # float16
        # seg_preds = ms.Tensor(seg_preds)

        sum_masks = sum_masks[keep]      # float32
        cate_scores = cate_scores[keep]  # float32
        cate_labels = cate_labels[keep]  # int


        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = ms.ops.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        
        # sort_inds = sort_inds.asnumpy().tolist()

        # seg_masks = seg_masks.asnumpy()
        seg_masks = seg_masks[sort_inds, :, :]
        # seg_masks = ms.Tensor(seg_masks)

        # seg_preds = seg_preds.asnumpy()
        seg_preds = seg_preds[sort_inds, :, :]
        # seg_preds = ms.Tensor(seg_preds)

        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        # print("G", seg_masks)

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg.kernel,sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        # breakpoint()
        temp = ms.ops.nonzero(keep).T[0]
        seg_preds = ms.ops.gather(seg_preds,temp,axis=0)
        # seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = ms.ops.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        
        idx = np.array([])

        seg_preds = seg_preds[sort_inds, :, :]
        
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores

    def loss(self,
             cate_preds,
             kernel_preds,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # import pdb;pdb.set_trace()
        mask_feat_size = ins_pred.shape[-2:]#torch_size(1, 256, 160, 240)
        # gt_bbox_list = [gt_bbox_list.data]#shape=[1, 4]
        # gt_label_list = [gt_label_list.data]#shape=[1]
        # gt_mask_list = [gt_mask_list.data]#(1, 640, 960)
        # gt_bbox_list = [gt_bbox_list]#shape=[1, 4]
        # gt_label_list = [gt_label_list]#shape=[1]
        # gt_mask_list = [gt_mask_list]#(1, 640, 960)
        #z30055003 去掉datacontainer 换成list
        if str(type(gt_bbox_list)) == "<class 'dataloader.data_container.DataContainer'>":
            gt_bbox_list = [gt_bbox_list.data]#shape=[1, 4]
        if str(type(gt_label_list)) == "<class 'dataloader.data_container.DataContainer'>":
            gt_label_list = [gt_label_list.data]#shape=[1]
        if str(type(gt_mask_list)) == "<class 'dataloader.data_container.DataContainer'>":
            gt_mask_list = [gt_mask_list.data]#(1, 640, 960)
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list, 
            mask_feat_size=mask_feat_size)
        #ins_label_list : [shape=[2, 160, 240],shape=[0],shape=[0],shape=[0],shape=[0]]
        #cate_label_list : [shape=[40, 40],shape=[36, 36],shape=[24, 24],shape=[16, 16],shape=[12, 12]]
        #ins_ind_label_list : [shape=[1600],shape=[1296],shape=[576],shape=[256],shape=[144]]
        #grid_order_list : [[[840, 880], [], [], [], []]]
        ins_labels = [mindspore.ops.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
        #ins_labels [Tensor(shape=[2, 160, 240],...]
        new_kernel_preds_result = []
        #[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list)):
            kernel_preds_level_result = []
            for kernel_preds_level_img, grid_orders_level_img in zip(kernel_preds_level, grid_orders_level):
                if len(grid_orders_level_img) == 0:
                    continue
                kernel_preds_level_img = kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                kernel_preds_level_result.append(kernel_preds_level_img)
            new_kernel_preds_result.append(kernel_preds_level_result)
        
        kernel_preds = new_kernel_preds_result
    
        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.shape[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape[0], kernel_pred.shape[1]
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                #z30055003 fp32 -> fp16
                if kernel_pred.dtype == mindspore.float16:
                    cur_ins_pred = mindspore.ops.conv2d(cur_ins_pred.astype(ms.float16), kernel_pred, stride=1).view(-1, H, W)
                else:
                    cur_ins_pred = mindspore.ops.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                #z30055003 ops.cat不支持bool类型
                b_mask_pred = ms.Tensor(np.concatenate([x.asnumpy() for x in b_mask_pred]))
                # b_mask_pred = mindspore.ops.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        # ins_ind_labels = [
        #     torch.cat([ins_ind_labels_level_img.flatten()
        #                for ins_ind_labels_level_img in ins_ind_labels_level])
        #     for ins_ind_labels_level in zip(*ins_ind_label_list)
        # ]# [torch.Size([1600]),torch.Size([1296]),torch.Size([576]),torch.Size([256]),torch.Size([144])] 2:shape乘2
        ins_ind_labels = [
            ms.Tensor(np.concatenate([ins_ind_labels_level_img.asnumpy().flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level]))
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        # flatten_ins_ind_labels = mindspore.ops.cat(ins_ind_labels)
        flatten_ins_ind_labels = ms.Tensor(np.concatenate([x.asnumpy() for x in ins_ind_labels]))
        # num_ins = flatten_ins_ind_labels.sum()
        num_ins = ms.ops.sum(flatten_ins_ind_labels)
 
        # dice loss
        loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):#target torch.Size([8, 192, 256]) torch.Size([8, 192, 256])
            if input is None:
                continue
            input = mindspore.ops.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = ms.Tensor(np.concatenate([x.asnumpy() for x in loss_ins])).mean()
        #loss_ins = mindspore.ops.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight
        
        # cate
        # cate_labels = [
        #     ms.Tensor(np.concatenate([cate_labels_level_img.asnumpy().flatten()
        #                for cate_labels_level_img in cate_labels_level]))
        #     for cate_labels_level in zip(*cate_label_list)
        # ]
        cate_labels = [
            mindspore.ops.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = mindspore.ops.cat(cate_labels)
 
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = mindspore.ops.cat(cate_preds)
 
        loss_cate_value = self.call_loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        logging.info(f"loss_ins:{loss_ins}, loss_cate:{loss_cate_value}")
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate_value)
 
    
    def solov2_target_single(self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, mask_feat_size):
        # logging.info(f'input: gt_bboxes_raw:{gt_bboxes_raw}\n gt_labels_raw:{gt_labels_raw}\n gt_masks_raw:{gt_masks_raw} mask_feat_size:{mask_feat_size}')
        # logging.info(f"mask_feat_size:{mask_feat_size}")
 
        # ins
        
        #z30055003 
        # ms.ops.
        # import pdb;pdb.set_trace()
        gt_areas = ms.ops.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))  
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):
            #z30055003
            # hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            hit_indices = ms.ops.logical_and((gt_areas >= lower_bound),(gt_areas <= upper_bound)).nonzero()
            if hit_indices.shape[0] != 0:
                hit_indices = hit_indices.flatten()
            
            num_ins = len(hit_indices)
            # logging.info(f'hit_indices:{hit_indices} {type(hit_indices)}\n num_ins:{num_ins}') #ok
            ins_label = []
            grid_order = []
            cate_label = ms.ops.zeros([num_grid, num_grid], dtype=ms.int64)
            ins_ind_label = ms.ops.zeros([num_grid ** 2], dtype=ms.bool_)
 
            if num_ins == 0:
                ins_label = ms.ops.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=ms.uint8)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.asnumpy(), ...]
            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma
 
            # mass center
            gt_masks_pt = ms.Tensor(gt_masks, dtype=ms.int64) #转换数据类型
            # import pdb;pdb.set_trace()
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            # logging.info(f'center_ws:{center_ws}, center_hs:{center_hs}')
            valid_mask_flags = gt_masks_pt.sum(axis=-1).sum(axis=-1) > 0
 
            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))
 
                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))
 
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)
 
                cate_label[top:(down+1), left:(right+1)] = gt_label
                #z30055003 使用opencv的操作，需要改成mindspore的操作
                seg_mask = mmcv_utils.imrescale(seg_mask, scale=1. / output_stride)  #todo
                # seg_mask = imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = ms.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
 
                        cur_ins_label = ms.ops.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=ms.uint8)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = ms.ops.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=ms.uint8)
            else:
                ins_label = ms.ops.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        # logging.info(f'output ins_label_list:{ins_label_list}\n cate_label_list:{cate_label_list}\n, ins_ind_label_list:{ins_ind_label_list}\n grid_order_list:{grid_order_list}')
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list
    
 
    def build_focal_loss(self, config):
        ms_s_focal_loss = SigmoidFoaclLoss(weight=None, gamma=config['gamma'], alpha=config['alpha'],
                                       reduction="mean")
        return ms_s_focal_loss
 
    def call_loss_cate(self, flatten_cate_preds, flatten_cate_labels, avg_factor):
        num_classes = flatten_cate_preds.shape[1]
        one_hot = mindspore.ops.OneHot()
        on_value = mindspore.Tensor(1.0, mindspore.float32)
        off_value = mindspore.Tensor(0.0, mindspore.float32)
        flatten_cate_labels = one_hot(flatten_cate_labels, num_classes + 1, on_value, off_value)
        flatten_cate_labels = flatten_cate_labels[:, 1:]
        loss = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor)
        return loss