from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore as ms
from mindspore import nn


class ClipGradientNorm(nn.Cell):
    """
    Clip gradients.
    """
 
    def __init__(self):
        super().__init__()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, grads, max_norm, norm_type:int = 2):
        if isinstance(grads, ms.Tensor):
            grads = [grads]
        max_norm = float(max_norm)
        norm_type = int(norm_type)
        if len(grads) == 0:
            return ms.tensor(0.)
        # not support inf norm
        grad_norms = [ops.norm(F.reshape(grad, (-1,)), dim=0, ord=norm_type, keepdim=False) for grad in grads]
        total_norm = ops.norm(F.stack(grad_norms, axis=0), dim=0, ord=norm_type, keepdim=False)
        
        clip_coef = max_norm / (total_norm + 1e-6)

        clip_coef_clamped = F.minimum(clip_coef, 1.0)

        new_grads = ()
        for grad in grads:
            t = grad * clip_coef_clamped
            new_grads = new_grads + (t,)
        return new_grads