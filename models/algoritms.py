import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import collections
import utils.commons as commons
import multiprocessing as mp
import os
import torch.cuda as cuda
from time import time
import json
import numpy as np
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'IGA',
    'SelfReg',
    'Epi_fcr',
    'g2dm'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, flags, hparams, input_shape, class_idx, class_balance):
        super(Algorithm, self).__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.class_balance = class_balance
        self.input_shape = input_shape
        self.class_idx = class_idx
        self.flags = flags
        self.num_classes = flags.num_classes
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.num_devices = cuda.device_count()
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')
        self.update_count = 0
        # A dictionnary containing all models
        self.setup()
        self.configure()

    def setup(self):
        raise NotImplementedError

    def configure(self):
        raise NotImplementedError

    def update(self, x, y, d):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x, y, d, *argv):
        raise NotImplementedError


