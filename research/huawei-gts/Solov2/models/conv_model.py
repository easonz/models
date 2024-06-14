""" network initialized related"""


import mindspore as ms
from typing import Union, Tuple
        
        
class Conv2D_Compatible_With_Torch(ms.nn.Cell):
    """
    A basic unit module for Conv2D.
    Interface remains the same as Conv2D.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias='auto',
        conv_cfg=None,
        norm_cfg=None,
        activation='relu',
        order=('conv', 'norm', 'act'),
        weight_init=None,
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias
        if conv_cfg == None:
            if padding == 0:
                self.conv = ms.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                      stride=stride, pad_mode='valid', padding=padding, dilation=dilation, group=groups, has_bias=bias,  bias_init='zeros', weight_init=weight_init)
            else:
                self.conv = ms.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                      stride=stride, pad_mode='pad', padding=padding, dilation=dilation, group=groups, has_bias=bias,  bias_init='zeros', weight_init=weight_init)
        else:
            self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            has_bias=bias)

        # if self.with_norm and self.with_bias:
        #     warnings.warn('ConvModule has norm and bias at the same time')

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            setattr(self, self.norm_name, norm)
            
            
        if self.with_activation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = ms.nn.ReLU()

    
    def construct(self, x):
        ori_type = x.dtype
        # x = x.astype(ms.float16)
        x = self.conv(x)
        # x = x.astype(ori_type)
        if self.with_norm:
            x = getattr(self, self.norm_name)(x)
        if self.with_activation:
            x = self.activate(x)
        return x


class ModulatedDeformConvPack(ms.nn.Cell):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 has_bias=True): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) == 2:        
                self.kernel_size = tuple(kernel_size)
            else:
                raise RuntimeError('kernel_size argument error')        
                        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, (list, tuple)):
            if len(stride) == 2:        
                self.stride = tuple(stride)
            else:
                raise RuntimeError('stride argument error')     
                        
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 2:        
                self.padding = (padding[0], padding[0], padding[1], padding[1])
            elif len(padding) == 4:
                self.padding = padding
            else:
                raise RuntimeError('argument error')
                        
                        
        self.dilation = dilation
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, (list, tuple)):
            if len(dilation) == 2:        
                self.dilation = tuple(dilation)
            else:
                raise RuntimeError('stride argument error')
                        
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = has_bias
        
        self.weight = ms.Parameter(
            ms.Tensor(shape=(out_channels, in_channels // groups, *self.kernel_size),
                      dtype=ms.float32,
                      init=ms.common.initializer.HeUniform()))
        if has_bias:
            self.bias = ms.Parameter(ms.Tensor([0.] * out_channels, dtype=ms.float32))
        else:
            self.bias = None
        # TODO：视情况而定看下是否做
        # self.reset_parameters()
        
        self.conv_offset = ms.nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            pad_mode = 'pad',
            dilation=self.dilation,
            group=self.groups,
            has_bias=True)
        
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.fill(value=0)
        self.conv_offset.bias.fill(value=0)
                        
    def construct(self, x):
        offset_with_mask = self.conv_offset(x)
        o1, o2, mask = ms.ops.chunk(offset_with_mask, 3, axis=1)                
        offset = ms.ops.concat((o1, o2), axis=1)
        mask = ms.ops.sigmoid(mask)
                        
        weight = self.weight
        strides = (self.stride[0], self.stride[1])
        bias = self.bias
        return deform_conv2d(x, offset, weight, bias=bias, stride=strides, padding=self.padding, 
        dilation=self.dilation, mask=mask)


conv_cfg = {
    'Conv': ms.nn.Conv2d,
    # 'ConvWS': ConvWS2d,
    # 'DCN': DeformConvPack,
    'DCNv2': ModulatedDeformConvPack,
    # TODO: octave conv
}

def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', ms.nn.BatchNorm2d),
    'SyncBN': ('bn', ms.nn.SyncBatchNorm),
    'GN': ('gn', ms.nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)


    return name, layer


def deform_conv2d(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), mask=None):
    ms_x = input
    ms_weight = weight

    batch = input.shape[0]
    kernel_height = weight.shape[-2]
    kernel_weight = weight.shape[-1]
    out_height = offset.shape[-2]
    out_width = offset.shape[-1]
    offset_groups = 1

    offset = offset.reshape((batch, offset_groups, kernel_height, kernel_weight, 2, out_height, out_width))
    offset_y, offset_x = ms.ops.chunk(offset, 2, axis=4)
    mask = mask.reshape((batch, offset_groups, kernel_height, kernel_weight, 1, out_height, out_width))

    ms_offsets = ms.ops.functional.concat([offset_x, offset_y, mask], axis=1)
    ms_offsets = ms_offsets.reshape(batch, 3 * offset_groups * kernel_height * kernel_weight, out_height, out_width)
    

    ms_kernel_size = tuple(weight.shape[-2:])
    
    ms_strides = (1, 1, stride[0], stride[1])
    ms_padding = (padding[0], padding[0], padding[1], padding[1])
    ms_bias = None
    ms_dilations = (1, 1, dilation[0], dilation[1])
    ms_groups = 1
    ms_deformable_groups = 1
    ms_modulated = True

    return ms.ops.deformable_conv2d(x=ms_x, weight=ms_weight, offsets=ms_offsets, kernel_size=ms_kernel_size, strides=ms_strides, padding=ms_padding,
        bias=ms_bias, dilations=ms_dilations, groups=ms_groups, deformable_groups=ms_deformable_groups, modulated=ms_modulated)