import numpy as np
import mindspore.common.initializer as initializer
from mindspore.common.initializer import *
from mindspore import Parameter
from .conv_model import *


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        module.weight = Parameter(initializer(XavierUniform(gain=gain), module.weight.shape), name=module.weight.name)
    else:
        module.weight = Parameter(initializer(XavierNormal(gain=gain), module.weight.shape), name=module.weight.name)
    if hasattr(module, 'has_bias') and getattr(module, 'has_bias'):
        module.bias = Parameter(initializer(Constant(module.bias, bias), module.bias.shape), name=module.bias.name)


def normal_init(module, mean=0, std=1, bias=0):
    if isinstance(module, Conv2D_Compatible_With_Torch):
        # breakpoint()
        module = module.conv
    module.weight = Parameter(initializer(Normal(std), module.weight.shape), name=module.weight.name)
    # breakpoint()
    if hasattr(module, 'has_bias') and getattr(module, 'has_bias'): 
        module.bias = Parameter(initializer(Constant(bias), module.bias.shape), name=module.bias.name)


def uniform_init(module, a=0, b=1, bias=0):
    module.weight = Parameter(initializer(Uniform( a, b)))
    if hasattr(module, 'has_bias') and getattr(module, 'has_bias'):
        module.bias = Parameter(initializer(Constant(module.bias, bias), module.bias.shape), name=module.bias.name)


def kaiming_init(module,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        # breakpoint()
        module.weight = Parameter(initializer(HeNormal(mode=mode, nonlinearity=nonlinearity), module.weight.shape), name=module.weight.name)
    else:
        # breakpoint()
        module.weight = Parameter(initializer(HeNormal(mode=mode, nonlinearity=nonlinearity), module.weight.shape), name=module.weight.name)
    if hasattr(module, 'has_bias') and getattr(module, 'has_bias'):
        module.bias = Parameter(initializer(Constant(bias), module.bias.shape), name=module.bias.name)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
