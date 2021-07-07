# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
import os
import time
import warnings
from itertools import islice

import numpy as np
import torch.nn.functional as F

try:
    from apex import amp
    AMP_AVAILABLE = True
except ModuleNotFoundError:
    AMP_AVAILABLE = False
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.video.resnet import VideoResNet

from .load_model import *
from .module_versions import  *

import torchvision

# from vu.utils import Config
# from vu.data import (
#     DEFAULT_MEAN,
#     DEFAULT_STD,
#     show_batch as _show_batch,
#     VideoDataset,
# )
#
# from vu.utils.metrics import accuracy, AverageMeter, retrieve_gt_pd, epoch_accuracy_fscore

# From https://github.com/moabitcoin/ig65m-pytorch
TORCH_R2PLUS1D = "moabitcoin/ig65m-pytorch"
MODELS = {
    # model: output classes
    'r2plus1d_34_32_ig65m': 359,
    'r2plus1d_34_32_kinetics': 400,
    'r2plus1d_34_8_ig65m': 487,
    'r2plus1d_34_8_kinetics': 400,
}

class R2Plus1D(object):
    def __init__(self, cfgs):
        self.configs = cfgs

        self.model = self.init_model(
            self.configs.sample_length,
            self.configs.base_model,
            self.configs.num_classes,
            self.configs.is_treble,
            self.configs.extended_version,
            self.configs.is_features
        )
        self.model_name = "r2plus1d_34_{}_{}".format(self.configs.sample_length, self.configs.base_model)


    @staticmethod
    def init_model(sample_length, base_model, num_classes=None, is_treble=False, extended_version = False, is_features = False):
        '''if sample_length not in (8, 32):
            raise ValueError(
                "Not supported input frame length {}. Should be 8 or 32"
                .format(sample_length)
            )'''
        if base_model not in ('ig65m', 'kinetics'):
            raise ValueError(
                "Not supported model {}. Should be 'ig65m' or 'kinetics'"
                    .format(base_model)
            )

        model_name = "r2plus1d_34_{}_{}".format(32, base_model)

        print("Loading {} model".format(model_name))

        model = torch.hub.load(
            TORCH_R2PLUS1D, model_name, num_classes=MODELS[model_name], pretrained=True
        )

        print('nombre de classe {:d}'.format(num_classes))

        # model.replace_logits(num_classes)

        # Replace head
        if num_classes is not None:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        #model_feat = MyModel(model)
        if is_treble:
            if not extended_version:
                model = MyTwoStreamModel(model)
            else:
                model = MyTwoStreamModelExtended(model)
        else:
            model = MyModel(model)



        return model



    def freeze(self):
        """Freeze model except the last layer"""
        self._set_requires_grad(False)
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        self._set_requires_grad(True)

    def _set_requires_grad(self, requires_grad=True):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def get_fc(self):
        return self.model.fc

