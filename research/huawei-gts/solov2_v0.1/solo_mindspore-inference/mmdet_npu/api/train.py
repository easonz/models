from collections import OrderedDict
import random
import re
import numpy as np
from mmdet_npu.utils import get_root_logger
from mindspore import nn
import numpy as np
import mindspore as ms
import sys
sys.path.append("../../")
from mmcv_npu import Runner
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # if deterministic:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        #z30055003 reduce操作 
        # if dist.is_available() and dist.is_initialized():
        #     loss_value = loss_value.data.clone()
        #     dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None):
    logger = get_root_logger(cfg.log_level)

    # start training
    # if distributed:
    #     _dist_train(
    #         model,
    #         dataset,
    #         cfg,
    #         validate=validate,
    #         logger=logger,
    #         timestamp=timestamp)
    # else:
    _non_dist_train(
        model,
        dataset,
        cfg,
        validate=validate,
        logger=logger,
        timestamp=timestamp)

def make_optimizer(model, optimizer_config:dict):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    # decay = set()
    # no_decay = set()
    # # loop over all modules / params
    # # all——model.trainable_params()
    
    # for param in model.trainable_params():
    #     param_name = param.name
    #     if param_name.endswith('bias'):
    #         # all biases will not be decayed
    #         no_decay.add(param_name)
    #     elif ('weight' in param_name) and ('norm' in param_name): 
    #         no_decay.add(param_name)
    #     elif ('pos_embed' in param_name) or ('cls_token' in param_name): 
    #         no_decay.add(param_name)
    #     else:
    #         decay.add(param_name)
    # #对所有可训练参数进行分组，将名字放入set中去重
    # pretrained = set()
    # no_pretrained = set()
    # # loop over all modules / params
    # for param in model.trainable_params():
    #     param_name = param.name
    #     if param_name.startswith('feature') and ('aggregate' not in param_name): 
    #         pretrained.add(param_name)
    #     else:
    #         no_pretrained.add(param_name)
    
    # # validate that we considered every parameter
    # param_set = {param.name for param in model.trainable_params()}
    # inter_params = decay & no_decay
    # union_params = decay | no_decay
    # if len(inter_params) != 0:
    #     raise RuntimeError("parameters %s made it into both decay/no_decay sets!" % (str(inter_params), ))
    # if len(param_set - union_params) != 0:
    #     raise RuntimeError("parameters %s were not separated into either decay/no_decay set!"  % (str(param_set - union_params), ))

    # inter_params = pretrained & no_pretrained
    # union_params = pretrained | no_pretrained
    # if len(inter_params) != 0:
    #     raise RuntimeError("parameters %s made it into both pretrained/no_pretrained sets!" % (str(inter_params), ))
    # if len(param_set - union_params) != 0:
    #     raise RuntimeError("parameters %s were not separated into either pretrained/no_pretrained set!" \
    #     % (str(param_set - union_params), ))
    # without warmup: call default schedulers
    # schedule_type = optimizer_config.get('schedule_type', None)
    # if schedule_type:
    #     max_epoch = optimizer_config["epochs"]
    #     num_iters_per_epoch = optimizer_config['num_iters_per_epoch']
    #     max_step = max_epoch * num_iters_per_epoch
    #     if schedule_type == "cosine":
    #         # step per iteration
    #         min_lr = optimizer_config.get("eta_min", 0.0)
    #         max_lr = learning_rate
    #         print('min_lr', min_lr)
    #         print('max_lr', max_lr)
    #         total_step = max_step
    #         step_per_epoch = 1
    #         decay_epoch = total_step // step_per_epoch
    #         learning_rate = np.array(nn.dynamic_lr.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)).astype(np.float32)
    #         learning_rate_div_100 = np.array(nn.dynamic_lr.cosine_decay_lr(min_lr, max_lr / 100, total_step, step_per_epoch, decay_epoch)).astype(np.float32)
    #         print(learning_rate[0], learning_rate[-1])
    #         print(learning_rate_div_100[0], learning_rate_div_100[-1])
    #     elif schedule_type == "multistep":
    #         # step every some epochs
    #         steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
    #         lr_list = [learning_rate, ] * steps[0]
    #         for i in range(len(steps) - 1):
    #             prev_lr = lr_list[-1]
    #             cur_lr = prev_lr * optimizer_config["gamma"]
    #             mile = steps[i + 1] - steps[i]
    #             lr_list.extend([cur_lr, ] * mile)
    #         if len(lr_list) < max_step:
    #             lr_list.extend([lr_list[-1] * optimizer_config["gamma"], ] * (max_step - len(lr_list)))
    #         learning_rate = ms.Tensor(np.array(lr_list).astype(np.float32))
    #         learning_rate_div_100 = learning_rate / 100
    #     else:
    #         raise TypeError("Unsupported scheduler!")
    optim_groups = []
    for param in model.trainable_params():
        optim_groups.append({
            "params":[param],
            "lr":optimizer_config['lr']
        })
    
    print('optim_groups:')
    for item in optim_groups:
        print(item)

    if optimizer_config["type"] == "SGD":
        optimizer = nn.SGD(
            optim_groups,
            learning_rate=optimizer_config['lr'],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = nn.AdamWeightDecay(
            optim_groups,
            learning_rate=optimizer_config['lr'],
            eps=1e-8, # pytorch version default
            weight_decay=optimizer_config['weight_decay'] # pytorch version default
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer
def _non_dist_train(model,
                    dataset,
                    cfg,
                    validate=False,
                    logger=None,
                    timestamp=None):
    if validate:
        raise NotImplementedError('Built-in validation is not implemented '
                                  'yet in not-distributed training. Use '
                                  'distributed training or test.py and '
                                  '*eval.py scripts instead.')
    # prepare data loaders
    # dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # print(dataset)
    # data_loaders = [
    #     build_dataloader(
    #         ds,
    #         cfg.data.imgs_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         cfg.gpus,
    #         dist=False) for ds in dataset
    # ]
    # put model on gpus
    # model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = make_optimizer(model, cfg.optimizer)
    import pdb; pdb.set_trace()
    for lr in optimizer.group_lr:
        print(lr)
    
    # import inspect
    # print(inspect.getfile(ms.nn.Optimizer))
    # optimizer = ms.nn.SGD(model.trainable_params(), 
    # learning_rate=cfg.optimizer.lr,
    # momentum=cfg.optimizer.momentum,
    # weight_decay=cfg.optimizer.weight_decay)
    
    
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, logger=logger)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    runner.grad_fn = grad_fn
    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    # runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    runner.run(dataset)