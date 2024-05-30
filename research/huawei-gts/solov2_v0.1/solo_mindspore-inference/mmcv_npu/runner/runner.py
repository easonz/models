import mindspore as ms
from ..hooks import Hook, LrUpdaterHook, OptimizerHook, lr_updater
import logging
from .priority import get_priority
class Runner(object):
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
                 batch_processor=None
                 ):
        # assert callable(batch_processor)
        
        # if optimizer is not None:
        #     self.optimizer = self.init_optimizer(optimizer)
        # else:
        #     self.optimizer = None
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

        def forward_fn(dataset):
            # import pdb;pdb.set_trace()
            outputs = self.batch_processor(self.model,dataset,train_mode=True)
            return outputs['loss']
            # logits = self.model(data)
            # loss = self.loss_fn(logits, label)
            # return loss, logits
        self.forward_fn = forward_fn
        #前向可以返回多个值 只对第一个return的值求梯度
        self.grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

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
    
    # def init_optimizer(self, optimizer):
    #     """Init the optimizer.
    #     待修改
    #     Args:
    #         optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
    #             optimizer object or a dict used for constructing the optimizer.

    #     Returns:
    #         :obj:`~torch.optim.Optimizer`: An optimizer object.

    #     Examples:
    #         >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
    #         >>> type(runner.init_optimizer(optimizer))
    #         <class 'torch.optim.sgd.SGD'>
    #     """
    #     # if isinstance(optimizer, dict):
    #     #     optimizer = obj_from_dict(optimizer, torch.optim,
    #     #                               dict(params=self.model.parameters()))
    #     # elif not isinstance(optimizer, torch.optim.Optimizer):
    #     #     raise TypeError(
    #     #         'optimizer must be either an Optimizer object or a dict, '
    #     #         'but got {}'.format(type(optimizer)))
    #     # return optimizer
    #     if isinstance(optimizer, dict):
    #         optimizer = obj_from_dict(optimizer, mindspore.nn.Optimizer
    #                                     dict(params=self.model.trainable_params()))
    #     elif not instance(optimizer, torch.optim.Optimizer):
    #         raise TypeError(
    #             'optimizer must be either an Optimizer object or a dict, '
    #             'but got {}'.format(type(optimizer)))
    #     return optimizer
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
                                optimizer_config=None
                                ):
        if optimizer_config is None:
            optimizer_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
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
        self.model.set_train()

        self.call_hook('before_train_epoch')
        size = len(dataset)
        for batch,data in enumerate(dataset):
            # for _key,_item in data:
            #     if _item.data != None:
                    
            #         if isinstance(_item,ms.Tensor):
            #             _item = _item.asnumpy()
            
            self.call_hook('before_train_iter')
            #通过self.grad_fn 计算loss与梯度
            # import pdb;pdb.set_trace()
            data[0]['img'] = data[0]['img'].data[0]
            data[0]['gt_bboxes'] = data[0]['gt_bboxes'].data[0]
            data[0]['gt_labels'] = data[0]['gt_labels'].data[0]
            data[0]['gt_masks'] = data[0]['gt_masks'].data[0]
            loss,grads = self.grad_fn(data[0])
            # (loss, _), grads = self.grad_fn(data[0])
            self.grads = grads
            
            if batch % 1 == 0:
                loss, current = loss.asnumpy(), self._iter
                print(f"loss: {loss:>7f} lr: {self.optimizer.lrs[0].value().asnumpy():>7f} [{current:>3d}/{size:>3d}] ")
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
    def run(self, dataset, workflow, max_epochs, **kwargs):
        
        #todo:workflow
        self._max_epochs = max_epochs

        self.call_hook('before_run')

        while self.epoch < max_epochs:
            self.train(dataset)

        self.call_hook("after_run")
        # size = dataset.get_dataset_size()
        # self.model.set_train()
        # for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        #     self.call_hook('before_train_iter')
        #     loss = train_step(data, label)
        #     if batch % 100 == 0:
        #         loss, current = loss.asnumpy(), batch
        #         print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
        #     self.call_hook('after_train_iter')
        # self.call_hook('after_run') 