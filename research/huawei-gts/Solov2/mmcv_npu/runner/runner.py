import os
import os.path as osp

import mindspore as ms
from mindspore import nn
from mindspore.train.serialization import save_checkpoint

from ..hooks import Hook, LrUpdaterHook, OptimizerHook, lr_updater, CheckpointHook
from .priority import get_priority
from ..hooks.clip_grad import ClipGradientNorm


class Runner(nn.TrainOneStepWithLossScaleCell):

    """A training helper for mindspore.

    Args:
        待修改
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """
    def __init__(self,
                 model,
                 optimizer=None,
                 batch_processor=None,
                 work_dir=None,
                 epoch=0,
                 iter=0
                 ):
        super().__init__(model, optimizer, ms.Tensor(1.0))
        # assert callable(batch_processor)

        # if optimizer is not None:
        #     self.optimizer = self.init_optimizer(optimizer)
        # else:
        #     self.optimizer = None
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self._hooks = []
        self._epoch = epoch
        self._iter = iter
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self._max_norm = 35
        self._norm_type = 2
        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir = osp.abspath(work_dir)
            if not os.path.exists(self.work_dir):
                os.makedirs(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        def forward_fn(dataset):

            outputs = self.batch_processor(self.model,dataset,train_mode=True)
            return outputs['loss']

        self.forward_fn = forward_fn

        self.grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
        self.grad_clip = ClipGradientNorm()

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None
                                ):
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def train(self, dataset):
        self.model.set_train(True)
        self.model.backbone.set_train(True)

        self.call_hook('before_train_epoch')
        size = len(dataset)
        for batch,data in enumerate(dataset):

            self.call_hook('before_train_iter')

            import time
            begintime = time.time()
            '''if data[0]['img_meta'].data[0][0]['filename'] != 'data/coco/train2017/000000391895.jpg':
                continue'''

            data[0]['img'] = data[0]['img'].data[0]
            data[0]['gt_bboxes'] = [ms.Tensor(item) for item in data[0]['gt_bboxes'].data[0]]
            data[0]['gt_labels'] = [ms.Tensor(item) for item in data[0]['gt_labels'].data[0]]
            data[0]['gt_masks'] = data[0]['gt_masks'].data[0]  #zjy


            loss,grads = self.grad_fn(data[0])

            grads = self.grad_clip(grads, self._max_norm, self._norm_type)
            grads = self.grad_reducer(grads)

            self.optimizer(grads)

            self.grads = grads
            # self.call_hook('after_train_iter')
            endtime = time.time()
            if batch % 1 == 0:
                loss, current = loss.asnumpy(), self._iter
                # print(f"epoch[{current//size}][{current%size}/{size:>3d}] loss: {loss:>7f} lr: {self.optimizer.lrs[0].value().asnumpy():>7f} [{current:>3d}/{size:>3d}], steptime: {endtime-begintime}")
                print(f"epoch[{self.epoch}/{self.max_epochs-1}], batch[{current%size}/{size-1}], tBatch[{current}], loss: {loss:>7f} lr: {self.optimizer.lrs[0].value().asnumpy():>7f} [{current:>3d}/{size:>3d}], steptime: {endtime-begintime}")


            self._iter += 1


        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, dataset, workflow, max_epochs, **kwargs):

        #todo:workflow
        self._max_epochs = max_epochs

        self.call_hook('before_run')

        while self.epoch < max_epochs:
            print(f'===[ReloadCKPT]==== Train at epoch: {self.epoch}, iter: {self.iter}')
            self.train(dataset)

        self.call_hook("after_run")

    def save_checkpoint(self,
                    out_dir,
                    filename_tmpl='epoch_{}_iter_{}_snapshot.ckpt',
                    save_optimizer=False,
                    meta=None,
                    create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1, self.iter)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath)
        print(f'===[ReloadCKPT]==== Save snapshot ckpt: {filepath}')
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            src = filename
            dst = osp.join(out_dir, 'latest.ckpt')
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
