import cv2
import logging
import numpy as np
import sys
import mindspore
from mindspore.dataset import ImageFolderDataset
from .mmcv_utils import *
from . import mmcv_utils
from .data_container import DataContainer as DC
import logging
class ResizeOperation(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert isinstance(self.img_scale, tuple)
            assert mmcv_utils.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        #np.random.seed(1)
        # assert isinstance(img_scales, tuple)
        assert mmcv_utils.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        # logging.info(f"scale_idx {scale_idx}, img_scale{img_scale}")

        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        # assert isinstance(img_scales, tuple) and len(img_scales) == 2
        assert mmcv_utils.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        # logging.info(f'_resize_img input:{results}\n scale {results["scale"]}, scale index {results["scale_idx"]}')
        if self.keep_ratio:
            # logging.info(results)
            img, scale_factor = mmcv_utils.imrescale(
                results['img'], results['scale'], return_scale=True)
            # logging.info(f'resize_img:{img}, scale_factor:{scale_factor}')
        else:
            img, w_scale, h_scale = mmcv_utils.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv_utils.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv_utils.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = np.stack(masks)

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv_utils.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv_utils.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results): #todo：入参的时候numpy, 在163拼起来成为dict
        # logging.info(f"Resize start:{results}")
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        # logging.info(f"Resize end:{results}")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


class RandomFlipOperation(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def __call__(self, results):
        # logging.info(f"RandomFlip start:{results}")
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv_utils.imflip(
                results['img'], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = np.stack([
                    mmcv_utils.imflip(mask, direction=results['flip_direction'])
                    for mask in results[key]
                ])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv_utils.imflip(
                    results[key], direction=results['flip_direction'])
        # logging.info(f'RandomFlip end:{results}')
        # logging.debug(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        # logging.info(f"Normalize start:{results}")
        results['img'] = mmcv_utils.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        # logging.info(f'Normalize end:{results}')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str

class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv_utils.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = mmcv_utils.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv_utils.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            if padded_masks:
                results[key] = np.stack(padded_masks, axis=0)
            else:
                results[key] = np.empty((0, ) + pad_shape, dtype=np.uint8)

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = mmcv_utils.impad(results[key], results['pad_shape'][:2])

    def __call__(self, results):
        # logging.info(f"Pad start:{results}")
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        # logging.info(f"Pad end:{results}")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, mindspore.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return mindspore.Tensor.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return mindspore.Tensor(data)
    elif isinstance(data, int):
        return mindspore.LongTensor([data])
    elif isinstance(data, float):
        return mindspore.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale, flip=False):
        self.transforms = transforms
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        assert mmcv_utils.is_list_of(self.img_scale, tuple)
        self.flip = flip


    def __call__(self, results):
        logging.debug(f'MultiScaleFlipAug call start, result.type{type(results)}, result:{results}')
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = flip
                data = self.__call_transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        logging.debug(f'MultiScaleFlipAug call end :{aug_data_dict}')
        return aug_data_dict

    def __call_transforms(self, results):
        for transform in self.transforms:
            results = transform(results)
        logging.debug(f'after transforms:{results.keys()},{results}')
        return results

class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """
    #breakpoint()
    def __call__(self, results):
        # logging.info(f"DefaultFormat start:{results}")
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(img, stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(results[key])
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                results['gt_semantic_seg'][None, ...], stack=True)
        # logging.info(f"DefaultFormat end:{results}")
        return results

    def __repr__(self):
        return self.__class__.__name__

# class DefaultFormatBundle(object):
#     """Default formatting bundle.

#     It simplifies the pipeline of formatting common fields, including "img",
#     "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
#     These fields are formatted as follows.

#     - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
#     - proposals: (1)to tensor, (2)to DataContainer
#     - gt_bboxes: (1)to tensor, (2)to DataContainer
#     - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
#     - gt_labels: (1)to tensor, (2)to DataContainer
#     - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
#     - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
#                        (3)to DataContainer (stack=True)
#     """
#     #breakpoint()
#     def __call__(self, results):
#         # logging.info(f"DefaultFormat start:{results}")
#         if 'img' in results:
#             img = results['img']
#             if len(img.shape) < 3:
#                 img = np.expand_dims(img, -1)
#             img = np.ascontiguousarray(img.transpose(2, 0, 1))
#             results['img'] = DC(to_tensor(img), stack=True)
#         for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
#             if key not in results:
#                 continue
#             results[key] = DC(to_tensor(results[key]))
#         if 'gt_masks' in results:
#             results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
#         if 'gt_semantic_seg' in results:
#             results['gt_semantic_seg'] = DC(
#                 to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
#         # logging.info(f"DefaultFormat end:{results}")
#         return results

#     def __repr__(self):
#         return self.__class__.__name__

class Collect(object):
    """
    Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        # logging.info(f"Collect start:{results}")
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_meta'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            if key in results:
                data[key] = results[key]
            else:
                data[key] = []
        # logging.info(f"Collect end:{data}")
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    multiScaleFlipAug = MultiScaleFlipAug(transforms=[ResizeOperation(keep_ratio=True), RandomFlipOperation(), Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        Pad(size_divisor=32), ImageToTensor(keys=['img'])], img_scale=(1333, 800))
    trans_vals = [multiScaleFlipAug]
    
    # results = {"img": "/mnt/denglian/coco/test2017/000000000001.jpg"}
    # for trans_val in trans_vals:
    #     results = trans_val(results)
    # print(results)

    dataset_val = ImageFolderDataset(dataset_dir="/mnt/denglian/coco/test_one/", decode=False)
    data_size = dataset_val.get_dataset_size()
    logging.info(f"data_size:{data_size}")
    # data_iter = dataset_val.create_dict_iterator()
    # for data in data_iter:
    #     logging.info(data)
    
        
    dataset_val = dataset_val.map(operations=trans_vals, input_columns=["image"])
    dataset_val = dataset_val.batch(1)
    data_iter = dataset_val.create_dict_iterator()
    for data in data_iter:
        logging.info(f'data:{data}')

if __name__ == "__main__":
    main()