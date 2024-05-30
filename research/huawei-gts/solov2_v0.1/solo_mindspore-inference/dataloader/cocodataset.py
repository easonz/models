import numpy as np
from pycocotools.coco import COCO
import os.path as osp
import pycocotools.mask as maskUtils
from functools import partial
from .collate_ms import collate
from .multi_scale_flip_aug import DefaultFormatBundle, Collect
import sys
from . import loading
import logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class CocoDataset():
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.load_files = loading.LoadImageFromFile()
        self.load_anno = loading.LoadAnnotations(with_bbox=True, with_mask=True) # according to config
        self.defaultformatbundle = DefaultFormatBundle()
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)

        self.img_infos = self.load_annotations(self.ann_file)

        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()


    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def __len__(self):
        return len(self.img_infos)


    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_test_img(idx)
            return data
        while True:
            data = self.prepare_train_img(idx)
            if data is None or 'gt_bboxes' not in data or 'gt_masks' not in data :
                logging.info(f'random because: gt_bboxes->{"gt_bboxes" not in data}  gt_masks->{"gt_mask" not in data}')
                idx = self._rand_another(idx)
                continue
            # dump_tensor(data['img'], r'/mnt/zjy/cocdataset_compare/ms_img.tensor')
            # dump_tensor(data['gt_bboxes'], r'/mnt/zjy/cocdataset_compare/ms_gt_bboxes.tensor')
            # dump_tensor(data['gt_labels'], r'/mnt/zjy/cocdataset_compare/ms_gt_labels.tensor')
            # import mindspore
            # dump_tensor(mindspore.Tensor(data['gt_masks']), r'/mnt/zjy/cocdataset_compare/ms_gt_mask.tensor')
            
            return data


    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        self.load_files(results)
        self.load_anno(results)
        # logging.info(f'origin input result: key:{len(results.keys())}, {results.keys()}\n {results}')
        tmp_dict = {'bbox_fields':['gt_bboxes_ignore','gt_bboxes' ]}
        for k, v in results.items():  #删除为空的value
            if k != 'img_info' and k != 'ann_info':
                if isinstance(v, np.ndarray) and v.size != 0:
                    tmp_dict[k] = v
                elif v:
                    tmp_dict[k] = v
        if 'gt_bboxes_ignore' not in tmp_dict:
            tmp_dict['bbox_fields'].remove('gt_bboxes_ignore')
        if 'gt_bboxes' not in tmp_dict:
            tmp_dict['bbox_fields'].remove('gt_bboxes')
        if 'gt_masks' not in tmp_dict:
            tmp_dict['mask_fields'] = []
        return tmp_dict


    def prepare_test_img(self, idx):  
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        self.load_files(results)
        return results


def test_train():
    # import sys
    # sys.path.append('/mnt/denglian/its')
    # import its_utils
    import mindspore.dataset as de
    import os
    import pickle
    data_root = '/mnt/denglian/coco/'
    ann_file = data_root + 'annotations/instances_val2017_two.json'
    img_prefix=data_root + 'val2017/'

    is_training = True
    cocodataset = CocoDataset(ann_file=ann_file, data_root=data_root, img_prefix=img_prefix, test_mode=False)
    dataset_column_names = ['res']
    ds = de.GeneratorDataset(cocodataset, column_names=dataset_column_names,
                            num_shards=None, shard_id=None,
                            num_parallel_workers=1, shuffle=is_training, num_samples=None)

    resize_ops = ResizeOperation([(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)], multiscale_mode='value', keep_ratio=True)
    randomflip_ops = RandomFlipOperation(flip_ratio=0.5)
    normal_ops = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    pad_ops = Pad(size_divisor=32)
    defaultformatbundle_ops = DefaultFormatBundle()
    collect_ops = Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
    train_ops = [resize_ops, randomflip_ops, normal_ops, pad_ops, defaultformatbundle_ops, collect_ops]

    dataset_train = ds.map(operations=train_ops, input_columns=['res'], output_columns=['res'])
    # dataset_train = dataset_train.batch(2, per_batch_map=collate)
    dataset_train = dataset_train.batch(2, per_batch_map=partial(collate, samples_per_gpu=1))
    data_iter = dataset_train.create_dict_iterator()
    for iter, item in enumerate(data_iter):
        logging.info(f'iter {iter} {item}')
        logging.info(f'img_meta {item["res"]["img_meta"].data}')
        logging.info(f'img {item["res"]["img"].data}')
        logging.info(f'gt_bboxes {item["res"]["gt_bboxes"].data}')
        logging.info(f'gt_labels {item["res"]["gt_labels"].data}')
        logging.info(f'gt_masks {item["res"]["gt_masks"].data}')


def test_val():
    # import sys
    # sys.path.append('/mnt/denglian/its')
    # import its_utils
    import mindspore.dataset as de
    import os
    data_root = r'/mnt/denglian/coco/'
    # data_root = '/home/zhangchenxi/tmp/SOLO/data/coco/'
    ann_file = data_root + 'annotations/instances_val2017.json'
    img_prefix=data_root + 'val2017/'
    is_training = False
    cocodataset = CocoDataset(ann_file=ann_file, data_root=data_root, img_prefix=img_prefix, test_mode=True)
    dataset_column_names = ['res']
    ds = de.GeneratorDataset(cocodataset, column_names=dataset_column_names,
                            num_shards=None, shard_id=None,
                            num_parallel_workers=1, shuffle=is_training, num_samples=None)

    multiScaleFlipAug = MultiScaleFlipAugOrigin(transforms=[ResizeOperation(keep_ratio=True), RandomFlipOperation(), Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32), ImageToTensor(keys=['img']),Collect(keys=['img'])], img_scale=(1333, 800))
    trans_vals = [multiScaleFlipAug]
    dataset_train = ds.map(operations=trans_vals, input_columns =['res'], output_columns=['res'])
    dataset_train = dataset_train.batch(2, per_batch_map=partial(collate, samples_per_gpu=1))
    data_iter = dataset_train.create_dict_iterator()
    for iter, item in enumerate(data_iter):
        logging.info(f'iter {iter} {item}')
        logging.info(f'img_meta {item["res"]["img_meta"]}')
        logging.info(f'img {item["res"]["img"]}')


if __name__ == '__main__':
    from multi_scale_flip_aug import *
    test_train()
    test_val()