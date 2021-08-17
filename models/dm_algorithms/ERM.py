from models import backbones
import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.cuda as cuda
from torch.autograd import Variable

from models.backbones.epi_fcr_backbones import resnet_vanilla, resnet_epi_fcr

from models.algoritms import Algorithm, DefaultModel
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp
import models.backbones.networks as networks 
from termcolor import colored, COLORS
import time
import collections
import json


import numpy as np




class ERM(Algorithm):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(ERM, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.datasets = datasets
        self.input_shape = input_shape
        self.flags = flags
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.checkpoint_path = checkpoint_path
        self.setup()
        self.setup_path()
        self.configure()
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')

    def setup(self):
        flags = self.flags
        commons.fix_all_seed(flags.seed)
        # Algorithms
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.network = self.network.cuda()      


    def configure(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    @property
    def factory(self):
        return {'network': self.network}
        
    def update(self, x, y, d):
        loss = F.cross_entropy(self.network(x), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
