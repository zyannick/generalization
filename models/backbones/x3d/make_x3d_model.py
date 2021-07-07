import os
import unittest

import torch
from .layers.swish import Swish
from .x3d import create_x3d, create_x3d_bottleneck_block
from torch import nn




def define_model(flags):

    dict_models = {
        'X3D-XS': (4, 160, 2.0, 2.2, 2.25),
        'X3D-S': (13, 160, 2.0, 2.2, 2.25),
        'X3D-M': (16, 224, 2.0, 2.2, 2.25),
        'X3D-L': (16, 312, 2.0, 5.0, 2.25)
    }

    model_pref = flags.model_pref

    input_clip_length, input_crop_size, width_factor, depth_factor, bottleneck_factor = dict_models[model_pref]

    model = create_x3d(
                input_clip_length=input_clip_length,
                input_crop_size=input_crop_size,
                model_num_class=400,
                dropout_rate=0.5,
                width_factor=width_factor,
                depth_factor=depth_factor,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                stem_dim_in=12,
                stem_conv_kernel_size=(5, 3, 3),
                stem_conv_stride=(1, 2, 2),
                stage_conv_kernel_size=((3, 3, 3),) * 4,
                stage_spatial_stride=(2, 2, 2, 2),
                stage_temporal_stride=(1, 1, 1, 1),
                bottleneck=create_x3d_bottleneck_block,
                bottleneck_factor=bottleneck_factor,
                se_ratio=0.0625,
                inner_act=Swish,
                head_dim_out=2048,
                head_pool_act=nn.ReLU,
                head_bn_lin5_on=False,
                head_activation=nn.Softmax,
            )


    return model