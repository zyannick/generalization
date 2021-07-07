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

from models.algoritms import DefaultModel
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp
import models.backbones.networks as networks 
from termcolor import colored, COLORS
import time
import collections
from models.dm_algorithms.ERM import erm


import numpy as np




class Mixup(erm.ERM):
    def __init__(self, flags, backbone,  num_domains, input_shape, datasets):
        super(Mixup, self).__init__(flags, backbone,  num_domains, input_shape, datasets)

        
    def train(self):

        flags = self.flags

        train_minibatches_iterator = zip(*self.train_dataloaders)


        self.network.train()
        self.network.bn_eval()
        self.best_accuracy_val = -1

        while self.cur_epoch < flags.n_epochs:

            step_start_time = time.time()

            minibatches_device = [(x.cuda(), y.cuda())
                for x,y in next(train_minibatches_iterator)]

            step_vals = self.update(minibatches_device)
            self.checkpoint_vals['step_time'].append(time.time() - step_start_time)

            for key, val in step_vals.items():
                self.checkpoint_vals[key].append(val)
            

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in commons.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}

    def predict(self, x):
        return self.network(x)
