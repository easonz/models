from .hook import Hook
from .clip_grad import ClipGradientNorm

class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip
        self.clip_grad_obj = ClipGradientNorm()

    def clip_grads(self, params):
        self.clip_grad_obj(
            list(filter(lambda p: p.requires_grad, params)), **self.grad_clip)

    def after_train_iter(self, runner):
        # pass
        # import pdb;pdb.set_trace()
        runner.optimizer(runner.grads)
        # runner.optimizer.zero_grad()
        # runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.trainable_params())
        # runner.optimizer.step()