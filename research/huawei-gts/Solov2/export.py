import os
import numpy as np
import mindspore
import mmcv
import cv2
import sys
import logging
import argparse
from models.solov2 import SOLOv2
from models.classes import get_classes
from dataloader.multi_scale_flip_aug import MultiScaleFlipAug, ResizeOperation, RandomFlipOperation, Normalize, Pad, ImageToTensor


def parse_args():
    parser = argparse.ArgumentParser(description='export mindir or onnx')
    parser.add_argument('--config', help='model config file path', required=True)
    parser.add_argument('--checkpoint', help='model checkpoint file path', required=True)
    parser.add_argument('--out_file_name', help='output file name', required=True)
    parser.add_argument('--out_file_format', choices=['onnx', 'mindir'], help='output file format', required=True)
    parser.add_argument('--deviceid', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0) #1 pynative, 0 graph
    args = parser.parse_args()
    
    return args

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


def build_model(args):
    if not os.path.exists(args.checkpoint):
        raise RuntimeError(f'checkpoint:{args.checkpoint} not exist')

    logging.info(f"load config file:{args.config} start")
    config = mmcv.Config.fromfile(args.config)
    logging.info(f"load config file:{args.config} success")
    config.model.pretrained = None

    model = build_from_cfg_ms(config.model, default_args=dict(train_cfg=None, test_cfg=config.test_cfg))
    model.CLASSES = get_classes('coco')
    model.cfg = config
    
    logging.info(f"load checkpoint:{args.checkpoint} start")
    mindspore.load_checkpoint(args.checkpoint, model)
    logging.info(f"load checkpoint:{args.checkpoint} success")
    for name, param in model.parameters_and_names():
        logging.debug(f"model parameters_and_names:{name}")
    return model


def build_inputs():
    img_path = "./demo.jpg"
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

    input_img = mindspore.ops.unsqueeze(mindspore.Tensor(results.pop('img')[0]), dim=0)
    input_img_meta = [results]

    logging.info(f"input_img.shape:{input_img.shape}")
    return (input_img, None, False)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    
    logging.info(f"run at device {args.deviceid}, mode:{args.mode}")
    mindspore.set_context(device_target='Ascend', device_id=args.deviceid, mode=args.mode, jit_syntax_level=mindspore.STRICT)
    
    model = build_model(args)
    inputs = build_inputs()

    if True:
        outputs = model(*inputs)
        for i in range(len(outputs)):
            logging.info(f"outputs[{i}] shape:{outputs[i].shape}, {outputs[i]}")
    else:
        logging.info(f"export start, file_name={args.out_file_name}, file_format:{args.out_file_format.upper()}")
        mindspore.export(model, *inputs, file_name=args.out_file_name, file_format=args.out_file_format.upper())
        logging.info("export success")


if __name__ == "__main__":
    main()