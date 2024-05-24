from __future__ import division
import argparse
import os
import os.path as osp
import time
import sys
from collections import OrderedDict

from dataloader import *
from dataloader.collate_ms import collate

from mmcv_npu import Config,mkdir_or_exist,Runner
from mmdet_npu import get_root_logger,train_detector

import mindspore
from mindspore.experimental import optim
import mindspore as ms
from mindspore.communication import init
from mindspore.communication import get_rank, get_group_size
from models.solov2 import SOLOv2

import logging
try:
    os.environ["MINDSPORE_HCCL_CONFIG_PATH"] = os.getenv("RANK_TABLE_FILE")
    rank_id = int(os.getenv("RANK_ID"))
    rank_size = int(os.getenv("RANK_SIZE"))
    device_id = int(os.getenv("DEVICE_ID"))
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()
    ms.set_seed(1)
    ms.set_context(device_id=device_id)

    print(f"rank_id: {rank_id}")
    print(f"device_id: {device_id}")
    print(f"rank_size: {rank_size}")

    print("distribute training...")
except TypeError:
    print("standalone training...")


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, ms.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):

            log_vars[loss_name] = loss_value[0].mean()
            for i in range(1,len(loss_value)):
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
    parser.add_argument('--config', default='configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py',
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


def build_dataset(cfg,cfg_data):
    ann_file = cfg['ann_file']
    img_prefix = cfg['img_prefix']

    is_training = True
    cocodataset = CocoDataset(ann_file=ann_file, data_root=None, img_prefix=img_prefix, test_mode=False)
    dataset_column_names = ['res']
    import mindspore.dataset as de
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
        num_parallel_workers = 2
    )
    #breakpoint()
    dataset_train = dataset_train.batch(cfg_data["imgs_per_gpu"], per_batch_map=collate, num_parallel_workers = 1)
    data_iter = dataset_train.create_dict_iterator()
    return dataset_train


def make_optimizer(model, optimizer_config:dict):
    optim_groups = []
    optim_groups.append({
        "params":[param for param in model.trainable_params()],
        "lr":optimizer_config['lr']
    })
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
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


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    mindspore.set_context(max_call_depth=10000)
    mindspore.set_context(mode=1)
    mindspore.set_context(device_target="Ascend")


    args = parse_args()
    cfg = Config.fromfile(args.config)

    #工作路径
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    #权重加载
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True


    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))

    dataset_train = build_dataset(cfg.data.train,cfg.data)

    model = build_from_cfg_ms(cfg.model, default_args=dict(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg))

    checkpoint_file = cfg.load_from
    try:
        checkpoint = ms.load_checkpoint(checkpoint_file, model)
    except:
        checkpoint = ms.load_checkpoint(checkpoint_file, model, choice_func=lambda x: not x.startswith("bbox_head.solo_cate"))

    optimizer = make_optimizer(model, cfg.optimizer)

    runner = Runner(model,optimizer, batch_processor, work_dir=cfg.work_dir)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config)
    

    runner.run(dataset_train,workflow=1,max_epochs=cfg.total_epochs)


if __name__ == '__main__':
    main()
