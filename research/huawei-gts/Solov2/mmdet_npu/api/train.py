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

    optimizer = make_optimizer(model, cfg.optimizer)
    import pdb; pdb.set_trace()
    for lr in optimizer.group_lr:
        print(lr)


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

    runner.run(dataset)