import inspect
import sys
import os
import logging
import argparse

import mmcv
import mindspore as ms 
import numpy as np
import cv2
from scipy import ndimage

from models.classes import get_classes
from dataloader.multi_scale_flip_aug import MultiScaleFlipAug, ResizeOperation, RandomFlipOperation, Normalize, Pad, ImageToTensor
from models.solov2 import SOLOv2



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


def show_result_ins(img_path,
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
    img = mmcv.imread(img_path)
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
        cur_mask_bool = cur_mask.astype(np.bool_)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]
        label_text = class_names[cur_cate]
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
    if out_file is None:
        return img_show
    else:
        logging.info(f"write {out_file}")
        mmcv.imwrite(img_show, out_file)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path', required=True)
    parser.add_argument('--checkpoint', help='checkpoint file', required=True)
    parser.add_argument('--dataroot', help='coco data root', default='./demo.jpg')
    parser.add_argument('--out', help='output result file', default='./demo_ms.jpg')
    parser.add_argument('--deviceid', help='run device id', default=0)
    args = parser.parse_args()
    return args

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    arg = parse_args()
    
    logging.info(f"run at device:{arg.deviceid}")
    ms.set_context(device_target='Ascend', device_id=arg.deviceid)
    
    # build the model from a config file and a checkpoint file
    config_file = arg.config
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = arg.checkpoint
    config = mmcv.Config.fromfile(config_file)
    config.model.pretrained = None
    logging.info('building model')
    model = build_from_cfg_ms(config.model, default_args=dict(train_cfg=None, test_cfg=config.test_cfg))
    logging.info('model init succeed')
    model.CLASSES = get_classes('coco')
    model.cfg = config
    logging.info(f'loading checkpoint from {checkpoint_file}')
    checkpoint = ms.load_checkpoint(checkpoint_file, model)
    logging.info('load checkpoint succeed')

    img_path = arg.dataroot
    multiScaleFlipAug = MultiScaleFlipAug(transforms=[ResizeOperation(keep_ratio=True), RandomFlipOperation(), Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            Pad(size_divisor=32), ImageToTensor(keys=['img'])], img_scale=(1333, 800))
    trans_vals = [multiScaleFlipAug]
    img = cv2.imread(img_path)
    
    results = {"img": img, 'img_shape': img.shape, 'ori_shape': img.shape}
    for trans_val in trans_vals:
        results = trans_val(results)
    
    img = ms.ops.unsqueeze(ms.Tensor(results.pop('img')[0]), dim=0)
    for key in results.keys():
        results.update({key : results.get(key)[0]})

    img_meta = []
    img_meta.append(results)
    

    logging.info(f'model forward start, img.type:{type(img)}, img:{img}, img_meta:{img_meta}')
    result = model.simple_test(img, img_meta)
    logging.info(f"model forward end, result.type:{type(result)}, {result}")

    show_result_ins(img_path, result, model.CLASSES, score_thr=0.25, out_file=arg.out)


if __name__ == "__main__":
    main()