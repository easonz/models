import logging
from mindspore import nn
import mindspore as ms
from .BaseDetector import BaseDetector
from .FPN_MindSpore import FPN_MindSpore
from .ResNet import ResNet
from .MaskFeatHead_ms import MaskFeatHead_ms
from .SOLOv2Head_ms import SOLOv2Head_ms


class SOLOv2(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLOv2, self).__init__()
        backbone_type = backbone.pop("type")
        self.backbone = ResNet(**backbone)

        if neck is not None:
            neck_type = neck.pop("type")
            self.neck = FPN_MindSpore(**neck)
        if mask_feat_head is not None:
            mask_feat_head_type = mask_feat_head.pop("type")
            self.mask_feat_head = MaskFeatHead_ms(**mask_feat_head)
        if bbox_head is not None:
            bbox_head_type = bbox_head.pop("type")
            self.bbox_head = SOLOv2Head_ms(**bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num = 0

    def init_weights(self, pretrained=None):
        super(SOLOv2, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.SequentialCell):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.SequentialCell):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg)
        # import pdb;pdb.set_trace()
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
    
        outs = self.bbox_head(x, eval=True)
        
        seg_inputs = []
        for i in range(len(outs)):
            seg_inputs.extend(outs[i])
       
        if self.with_mask_feat_head:
            
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)

        else:
            seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result  

    def forward_mindir(self, img):
        x = self.extract_feat(img)
        
        outs = self.bbox_head(x, eval=True)
        
        seg_inputs = []
        for i in range(len(outs)):
            seg_inputs.extend(outs[i])
        
        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
            seg_inputs.append(mask_feat_pred)
    
        return seg_inputs  
    
    
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError