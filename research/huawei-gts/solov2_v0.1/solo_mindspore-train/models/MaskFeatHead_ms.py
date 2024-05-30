import mindspore as ms
from .conv_model import Conv2D_Compatible_With_Torch
from mindspore.ops import functional as F
from .weight_init_utils import *


class MaskFeatHead_ms(ms.nn.Cell):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None):
        super().__init__(auto_prefix=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.convs_all_levels = ms.nn.CellList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = ms.nn.SequentialCell()
            if i == 0:
                one_conv = Conv2D_Compatible_With_Torch(
                    self.in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
                convs_per_level.append(one_conv)
                
                self.convs_all_levels.append(convs_per_level)
                
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = Conv2D_Compatible_With_Torch(
                        chn,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
                    convs_per_level.append(one_conv)
                    
                    one_upsample = ms.nn.Upsample(
                        scale_factor=2.0, mode='bilinear', align_corners=False,recompute_scale_factor=True)
                    convs_per_level.append(one_upsample)
                    continue

                one_conv = Conv2D_Compatible_With_Torch(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
                    # inplace=False)
                convs_per_level.append(one_conv)
                
                one_upsample = ms.nn.Upsample(
                    scale_factor=2.0,
                    mode='bilinear',
                    align_corners=False,
                    recompute_scale_factor=True)
                convs_per_level.append(one_upsample)
                
            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ms.nn.SequentialCell(
            Conv2D_Compatible_With_Torch(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg),
        )

    def init_weights(self):
        for m in self.cells():
            if isinstance(m, ms.nn.Conv2d):
                normal_init(m, std=0.01)

        
    def construct(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0].astype(ms.float32))
        for i in range(1, len(inputs)):
            input_p = inputs[i].astype(ms.float32)
            if i == 3:
                input_feat = input_p
                x_range = ms.numpy.linspace(-1, 1, input_feat.shape[-1])
                y_range = ms.numpy.linspace(-1, 1, input_feat.shape[-2])
                y, x = ms.numpy.meshgrid(y_range, x_range)
                x = x.T
                y = y.T
                y = F.broadcast_to(y, tuple([input_feat.shape[0], 1, -1, -1]))
                x = F.broadcast_to(x, tuple([input_feat.shape[0], 1, -1, -1]))
                coord_feat = ms.ops.cat([x, y], 1)
                ori_type0 = coord_feat.dtype
                ori_type1 = input_p.dtype
                # coord_feat = coord_feat.astype(ms.float16)
                # input_p = input_p.astype(ms.float16)
                input_p = ms.ops.cat([input_p, coord_feat], 1)
                input_p = input_p.astype(ori_type1)
                coord_feat = coord_feat.astype(ori_type0)
            # print("******************mask113:",input_p.dtype)
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)
            

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred