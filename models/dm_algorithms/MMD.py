from models import backbones
import os

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import copy
from models.dm_algorithms.ERM import ERM
import utils.commons as commons
from .AbstractMMD import  AbstractMMD


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(MMD, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance, gaussian=True)
