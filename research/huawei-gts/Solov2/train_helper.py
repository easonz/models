from __future__ import division

import argparse
import os
from collections import OrderedDict

import mindspore as ms
import mindspore.dataset as de
from mindspore.communication import get_rank, get_group_size
from mindspore.experimental import optim

from dataloader import CocoDataset, ResizeOperation, RandomFlipOperation, Normalize, Pad, DefaultFormatBundle, Collect
from dataloader.collate_ms import collate
from models.solov2 import SOLOv2


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, ms.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = loss_value[0].mean()
            for i in range(1, len(loss_value)):
                log_vars[loss_name] = log_vars[loss_name] + loss_value[i].mean()
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = None
    for _key, _value in log_vars.items():
        if 'loss' in _key:
            if loss == None:
                loss = _value
            else:
                loss = loss + _value
    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img']))

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='./configs/solov2/solov2_r101_fpn_8gpu_3x.py',
                        help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def build_dataset(cfg, cfg_data):
    ann_file = cfg['ann_file']
    img_prefix = cfg['img_prefix']

    is_training = True
    cocodataset = CocoDataset(ann_file=ann_file, data_root=None, img_prefix=img_prefix, test_mode=False)
    dataset_column_names = ['res']

    try:
        num_shards = get_group_size()
        shard_id = get_rank()
    except:
        num_shards = None
        shard_id = None
    ds = de.GeneratorDataset(cocodataset, column_names=dataset_column_names,
                             num_shards=num_shards, shard_id=shard_id,
                             num_parallel_workers=2, shuffle=is_training, num_samples=None)
    train_ops = []
    for ops in cfg['pipeline']:
        if ops['type'] == 'Resize':
            train_ops.append(ResizeOperation(
                img_scale=ops['img_scale'],
                multiscale_mode=ops['multiscale_mode'],
                keep_ratio=ops['keep_ratio']
            ))
        if ops['type'] == 'RandomFlip':
            train_ops.append(RandomFlipOperation(
                flip_ratio=ops['flip_ratio']
            ))
        if ops['type'] == 'Normalize':
            train_ops.append(Normalize(
                mean=ops['mean'],
                std=ops['std'],
                to_rgb=ops['to_rgb']
            ))
        if ops['type'] == 'Pad':
            train_ops.append(Pad(
                size_divisor=ops['size_divisor']
            ))
        if ops['type'] == 'DefaultFormatBundle':
            train_ops.append(DefaultFormatBundle())
        if ops['type'] == 'Collect':
            train_ops.append(Collect(
                keys=ops['keys']
            ))
    dataset_train = ds.map(
        operations=train_ops,
        input_columns=['res'],
        output_columns=['res'],
        num_parallel_workers=2
    )

    dataset_train = dataset_train.batch(cfg_data["imgs_per_gpu"], per_batch_map=collate, num_parallel_workers=1,
                                        drop_remainder=True)
    data_iter = dataset_train.create_dict_iterator()
    return dataset_train


def make_optimizer(model, optimizer_config: dict):
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            [param for param in model.trainable_params() if param.requires_grad],
            lr=optimizer_config['lr'],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise TypeError("Unsupported optimizer!")
    return optimizer


def build_from_cfg_ms(cfg, default_args=None):
    args = cfg.copy()
    obj_type = args.pop('type')
    obj_cls = SOLOv2

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)
