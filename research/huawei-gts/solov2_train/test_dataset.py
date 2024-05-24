import inspect
import sys
import os
import logging
import argparse
import time

import mmcv
import mindspore as ms 
import numpy as np
import cv2
from scipy import ndimage
import pycocotools.mask as mask_util

from dataloader import *
from dataloader.collate_ms import collate
from mmcv_npu import Config,mkdir_or_exist,Runner
from mmdet_npu import get_root_logger,train_detector
from models.classes import get_classes
from dataloader.multi_scale_flip_aug import MultiScaleFlipAug, ResizeOperation, RandomFlipOperation, Normalize, Pad, ImageToTensor
from models.solov2 import SOLOv2
from dataloader.coco_utils import coco_eval, results2json_segm

# ms.set_context(device_target='Ascend', device_id=5, mode=ms.GRAPH_MODE)
ms.set_context(device_target='Ascend', device_id=3)
# ms.set_context(device_target='CPU')

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
    # breakpoint()
    logging.info("starting making output img")
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    if not result or result == [None]:

        logging.info("unsucessefully end here")
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
        #label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
    if out_file is None:
        logging.info(f"empty out_file path")
        return img_show
    else:
        logging.info(f"write {out_file}")
        mmcv.imwrite(img_show, out_file)


def build_dataset(cfg,cfg_data):
    ann_file = cfg['ann_file']
    img_prefix = cfg['img_prefix']

    is_training = False
    cocodataset = CocoDataset(ann_file=ann_file, data_root=None, img_prefix=img_prefix, test_mode=False)
    dataset_column_names = ['res']
    import mindspore.dataset as de
    ds = de.GeneratorDataset(cocodataset, column_names=dataset_column_names,
                        num_shards=None, shard_id=None,
                        num_parallel_workers=4, shuffle=is_training, num_samples=None)
    test_ops = []
    # breakpoint()
    for ops in cfg['pipeline']:
        if ops['type'] == 'MultiScaleFlipAug':
            test_ops.append(
            MultiScaleFlipAug(transforms=[ResizeOperation(keep_ratio=True), 
            RandomFlipOperation(), 
            Normalize(mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375], 
            to_rgb=True),
            Pad(size_divisor=32), ImageToTensor(keys=['img']), Collect(keys=['img'])], img_scale=(1333, 800),flip=False))

    dataset_train = ds.map(
        operations=test_ops, 
        input_columns=['res'], 
        output_columns=['res'],
        num_parallel_workers = 1
    )
    dataset_train = dataset_train.batch(cfg_data["imgs_per_gpu"], per_batch_map=collate, num_parallel_workers=1)
    data_iter = dataset_train.create_dict_iterator()
    return dataset_train

def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].numpy().astype(np.uint8)
        cate_label = cur_result[1].numpy().astype(np.int_)
        cate_score = cur_result[2].numpy().astype(np.float_)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)

        return masks

def single_gpu_test(model, data_loader, show=False, verbose=True):
    results = []
    dataset = data_loader
    num_classes = 81
    avg_time = np.array([])
    # prog_bar = mmcv.ProgressBar(len(dataset))
    prog_bar = mmcv.ProgressBar(10)
    i = 0
    for i, data in enumerate(dataset):
        if i == 10:
            break
        # breakpoint()
        data = data[0]
        data['img_meta'][0] = data['img_meta'][0].data[0]
        start_time = time.time()
        seg_result = model(return_loss=False, rescale=True, **data)
        # seg_result = model.simple_test()
        avg_time = np.append(avg_time, (time.time() - start_time))
        # logging.info(f'time cost: {float(time.time() - start_time) * 1000}ms')
        # breakpoint()
        result = {'result' : seg_result, 'img_meta': data['img_meta'][0]}
        # result = {'result' : get_masks(seg_result, num_classes=num_classes), 'img_meta': data['img_meta'][0].data[0]}
        results.append(result)
        
        # breakpoint()
        batch_size = data['img'][0].shape[0]
        for _ in range(batch_size):
            prog_bar.update()
        i+=1
    print()
    logging.info(f'avg time cost: {float(np.mean(np.delete(avg_time, np.argmax(avg_time)))) * 1000}ms')
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('dataroot', help='coco data root')
    parser.add_argument('--out', help='output result file', default='./out_put/ms.pkl')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    arg = parse_args()

    # build the model from a config file and a checkpoint file
    config_file = arg.config
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = arg.checkpoint
    cfg = Config.fromfile(config_file)
    config = cfg.copy()
    config.model.pretrained = None
    logging.info('building model')
    model = build_from_cfg_ms(config.model, default_args=dict(train_cfg=None, test_cfg=config.test_cfg))
    logging.info('model init succeed')
    model.CLASSES = get_classes('coco')
    model.cfg = config
    logging.info(f'loading checkpoint from {checkpoint_file}')
    checkpoint = ms.load_checkpoint(checkpoint_file, model)
    logging.info('load checkpoint succeed')

    path = arg.dataroot
    files = os.listdir(path+'/val2017')
    import time
    
    config.data_root = arg.dataroot
    dataset_test = build_dataset(config.data.test,config.data)
    outputs = single_gpu_test(model, dataset_test)
    # breakpoint()
    # mmcv.dump(outputs, arg.out)
    for item in outputs :
        show_result_ins(item['img_meta'][0]['filename'], item['result'], model.CLASSES, score_thr=0.25, out_file=str('./output/ms_' + item['img_meta'][0]['filename'].split('/')[-1]))
    logging.info('done here')

if __name__ == "__main__":
    main()