from __future__ import division

import logging
import os.path as osp
import sys
import time

import mindspore
import mindspore as ms
from mindspore.communication import init

from mmcv_npu import Config, mkdir_or_exist, Runner
from mmdet_npu import get_root_logger
from train_helper import parse_args, build_from_cfg_ms, make_optimizer, batch_processor, build_dataset

try:
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()
    ms.set_seed(1)
    print("distribute training...")
except TypeError:
    print("standalone training...")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(process)d] [%(thread)d] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    mindspore.set_context(max_call_depth=10000)
    mindspore.set_context(mode=1)
    mindspore.set_context(device_target="Ascend")

    args = parse_args()
    cfg = Config.fromfile(args.config)

    # 工作路径
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # 权重加载
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))

    # nit the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info('Config:\n{}'.format(cfg.text))

    dataset_train = build_dataset(cfg.data.train, cfg.data)

    model = build_from_cfg_ms(cfg.model, default_args=dict(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg))

    checkpoint_file = cfg.load_from

    try:
        ms.load_checkpoint(checkpoint_file, model)
    except:
        ms.load_checkpoint(checkpoint_file, model, choice_func=lambda x: not x.startswith("bbox_head.solo_cate"))

    optimizer = make_optimizer(model, cfg.optimizer)

    runner = Runner(model, optimizer, batch_processor, work_dir=cfg.work_dir)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config)
    runner.run(dataset_train, workflow=1, max_epochs=cfg.total_epochs)


if __name__ == '__main__':
    main()
