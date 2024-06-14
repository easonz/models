# Copyright (c) Open-MMLab. All rights reserved.
import collections
from .data_container import DataContainer
import mindspore
import numpy as np


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, np.ndarray):
        ret = mindspore.Tensor(np.stack([i for i in batch], 0))

        return ret
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([mindspore.Tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return mindspore.Tensor(batch)
    elif isinstance(elem, float):
        return mindspore.Tensor(batch, dtype=mindspore.float64)
    elif isinstance(elem, int):
        return mindspore.Tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_impl(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):

        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:

            for i in range(0, len(batch), samples_per_gpu):

                assert isinstance(batch[i].data, np.ndarray)
                if batch[i].pad_dims is not None:

                    if isinstance(batch[i].data, np.ndarray):
                        ndim = batch[i].data.ndim
                    else:
                        ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].data.shape[-dim]
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].data.shape[dim] == sample.data.shape[dim]
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.data.shape[-dim])
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.data.shape[-dim]

                        datalen = len(sample.data.shape)
                        npad = int(len(pad)/2)
                        new_pad = [(0,0)]*datalen
                        for i in range(npad):
                            new_pad[-(i+1)] = pad[2*i],pad[2*i+1]
                        padded_samples.append(np.pad(sample.data, new_pad, 'constant',
                        constant_values=(sample.padding_value,sample.padding_value)))

                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')
        else:

            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_impl(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {
            key: collate_impl([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def collate(batch, batch_info):
    return collate_impl(batch, len(batch))