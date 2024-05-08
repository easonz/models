import inspect
import argparse
import mmcv
import os
import mindspore as ms 
import numpy as np
import cv2
from scipy import ndimage
import mindspore.dataset
from models.classes import get_classes
from dataloader.multi_scale_flip_aug import MultiScaleFlipAug, ResizeOperation, RandomFlipOperation, Normalize, Pad, ImageToTensor
from models.solov2 import SOLOv2
from dataloader.cocodataset import CocoDataset
import logging
import sys
import pickle
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_from_cfg_ms(cfg, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        obj_cls = SOLOv2
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)


def show_result_ins(img,
                    result,
                    class_names,
                    score_thr=0.3,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """

    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    if not result or result == [None]:
        return img_show
    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.numpy()
    score = cur_result[2].numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label_text = class_names[cur_cate]
        #label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
    if out_file is None:
        return img_show
    else:
        mmcv.imwrite(img_show, out_file)


def build_val_dataset(data_root):
    ann_file = os.path.join(data_root, "annotations/instances_val2017.json")
    img_prefix = os.path.join(data_root, "val2017")
    print(f"ann_file:{ann_file}, data_root:{data_root}, img_prefix:{img_prefix}")

    coco_dataset = CocoDataset(ann_file=ann_file, data_root=data_root, img_prefix=img_prefix, test_mode=True)
    # dataset_column_names = ['res']
    # ds = mindspore.dataset.GeneratorDataset(cocodataset, column_names=dataset_column_names,
    #                         num_shards=None, shard_id=None,
    #                         num_parallel_workers=1, shuffle=False, num_samples=None)

    return coco_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('dataroot', help='coco data root')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].numpy().astype(np.uint8)
        cate_label = cur_result[1].numpy().astype(np.int32)
        cate_score = cur_result[2].numpy().astype(np.float64)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)

        return masks

def segm2json_segm(coco, results):
    segm_json_results = []
    for idx in range(len(results)):
        img_id = coco.getImgIds()[idx]
        seg = results[idx]
        for label in range(len(seg)):
            masks = seg[label]
            for i in range(len(masks)):
                mask_score = masks[i][1]
                segm = masks[i][0]
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score)
                data['category_id'] = coco.getCatIds()[label]
                segm['counts'] = segm['counts'].decode()
                data['segmentation'] = segm
                segm_json_results.append(data)
    return segm_json_results

#python val.py /mnt/denglian/SOLO/configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py ./SOLOv2_R101_DCN_3x-fpn.ckpt /mnt/denglian/coco --show --out  results_solo.pkl --eval segm
def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # build the model from a config file and a checkpoint file
    arg = parse_args()

    config = mmcv.Config.fromfile(arg.config)
    config.model.pretrained = None

    logging.debug("build model start")
    model = build_from_cfg_ms(config.model, default_args=dict(train_cfg=None, test_cfg=config.test_cfg))
    logging.debug("build model success")

    model.CLASSES = get_classes('coco')
    model.cfg = config
    num_classes = len(model.CLASSES)

    logging.debug("load checkpoint start")
    checkpoint = ms.load_checkpoint(arg.checkpoint, model)
    logging.debug("load checkpoint success")
    
    logging.debug("build val dataset start")
    dataset_val = build_val_dataset(arg.dataroot)
    logging.debug("build val dataset success")

    #data_iter = dataset_val.create_dict_iterator()
    multiScaleFlipAug = MultiScaleFlipAug(transforms=[ResizeOperation(keep_ratio=True), RandomFlipOperation(), Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32), ImageToTensor(keys=['img'])], img_scale=(1333, 800))
    trans_vals = [multiScaleFlipAug]

    results = []
    for i, data in enumerate(dataset_val):
        logging.info(f"data:{data}")
        for trans_val in trans_vals:
            data = trans_val(data)

        img = ms.ops.unsqueeze(data["img"][0], dim=0)
        logging.debug(img)
        img_meta = []
        img_meta.append(data)
    
        logging.debug(f'model forward start, img.type:{type(img)}, img:{img}, img_meta:{img_meta}')
        seg_result = model.simple_test(img, img_meta)
        logging.debug(f"model forward end, seg_result.type:{type(seg_result)}, {seg_result}")

        result = get_masks(seg_result, num_classes=num_classes)
        results.append(result)


    annotation_file = os.path.join(arg.dataroot, "annotations/instances_val2017.json")
    gt = COCO(annotation_file)
    dt_json = segm2json_segm(gt, results)
    coco_dets = gt.loadRes(dt_json)
    cocoEval = COCOeval(gt, coco_dets, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    main()