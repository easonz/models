import mindspore as ms
from mindspore import nn
from mindspore.ops import functional as F
from .hook import Hook
from .clip_grad import ClipGradientNorm
# import logging

class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip
        self.clip_grad_obj = ClipGradientNorm()
        self.grad_reducer = F.identity
        self.parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        self.reducer_flag = False
        if self.parallel_mode in [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        

    def clip_grads(self, params):
        self.clip_grad_obj(
            list(filter(lambda p: p.requires_grad, params)), **self.grad_clip)
    
    def reducer(self, grads, params):
        mean = ms.context.get_auto_parallel_context("gradients_mean")
        degree = ms.context.get_auto_parallel_context("device_num")
        self.grad_reducer = nn.DistributedGradReducer(params, mean, degree)
        return 

    def after_train_iter(self, runner):

        if self.grad_clip is not None:
            runner.grads = self.clip_grad_obj(runner.grads, self.grad_clip["max_norm"], self.grad_clip["norm_type"])
            if self.reducer_flag:
                runner.grads = self.reducer(runner.grads, runner.model.get_parameters())
            else:
                runner.grads = self.grad_reducer(runner.grads)
        runner.optimizer(runner.grads)





