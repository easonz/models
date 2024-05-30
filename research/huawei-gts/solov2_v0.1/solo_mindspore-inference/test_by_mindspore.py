#import os
#import numpy as np
import mindspore
import mindspore.dataset as ds
import os
import numpy as np
import mmcv
import cv2
import sys
import logging
import argparse
import cv2
import pickle
from models.solov2 import SOLOv2
from models.classes import get_classes
from dataloader.multi_scale_flip_aug import MultiScaleFlipAug, ResizeOperation, RandomFlipOperation, Normalize, Pad, ImageToTensor
from mmdet_npu.api import show_result_ins

def parse_args():
    parser = argparse.ArgumentParser(description='test by mindir')
    parser.add_argument('--config', help='test config file path', default='./configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py')
    parser.add_argument('--mindir', help='mindir', required=True)
    parser.add_argument('--dataroot', help='coco data root', default='./demo.jpg')
    parser.add_argument('--out', help='output result file', default='./demo_mslite_result.jpg')
    args = parser.parse_args()
    
    return args


def build_model_by_mindspore(args):
    logging.info(f"mindspore.load start, {args.mindir}")
    graph = mindspore.load(args.mindir)
    model = mindspore.nn.GraphCell(graph)
    logging.info(f"mindspore.load end")
    
    return model


def build_inputs(img_path):
    multiScaleFlipAug = MultiScaleFlipAug(transforms=[ResizeOperation(keep_ratio=True), RandomFlipOperation(),
                                                    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                                                                to_rgb=True),
                                                    Pad(size_divisor=32), ImageToTensor(keys=['img'])],
                                        img_scale=(1333, 800))
    trans_vals = [multiScaleFlipAug]
    img = cv2.imread(img_path)
    results = {"img": img, 'img_shape': img.shape, 'ori_shape': img.shape}
    for trans_val in trans_vals:
        results = trans_val(results)

    logging.info(f"results:{results}")
    input_img = mindspore.ops.unsqueeze(mindspore.Tensor(results.pop('img')[0]), dim=0)
    input_img_meta = [results]

    with open("input_img.pkl", "wb") as fd:
        pickle.dump(input_img.numpy(), fd)
        
    logging.info(f"input_img.shape:{input_img.shape}")
    return (input_img, input_img_meta)

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


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target='Ascend', device_id=6, save_graphs=3, save_graphs_path="./load_graph")

    
    model = build_model_by_mindspore(args)
    
    input_img, input_img_meta = build_inputs(args.dataroot)
    
    logging.info(f"model predict start, input_img.shape:{input_img.shape}")
    outputs = model(input_img)
    logging.info(f"model predict success, outputs.len:{len(outputs)}")
    for i in range(len(outputs)):
        logging.info(f"outputs[{i}] shape:{outputs[i].shape}, {outputs[i]}")



if __name__ == "__main__":
    main()