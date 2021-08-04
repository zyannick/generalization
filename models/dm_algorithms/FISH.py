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



import numpy as np




class Fish(DefaultModel):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(Fish, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.datasets = datasets
        # self.backbone = gears['backbone']
        # self.discriminator = gears['disc']
        # self.classifier = gears['classifier']
        self.input_shape = input_shape
        self.flags = flags
        self.setup()
        self.setup_path()
        self.configure()

    def setup(self):
        flags = self.flags
        commons.fix_all_seed(flags.seed)
        # Algorithms
        self.network = networks.WholeFish(self.input_shape, flags.num_classes, self.hparams)
        self.network = self.network.cuda()
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)


    def configure(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None




    def load_checkpoint(self, epoch):

        ckpt = self.save_epoch_fmt_task.format(epoch)

        ck_name = ckpt

        if os.path.isfile(ckpt):

            ckpt = torch.load(ckpt)
            # Load model state
            self.network.load_state_dict(ckpt['network'])

            # Load history
            self.history = ckpt['history']
            self.cur_epoch = ckpt['cur_epoch']
            self.current_lr = ckpt['lr']

            print('Checkpoint number {} loaded  ---> {}'.format(
                epoch, ck_name))
            return True
        else:
            print(colored('No checkpoint found at: {}'.format(ckpt), 'red'))
            if self.flags.phase != 'train':
                raise ValueError(
                    '----------Unable to load checkpoint  {}. The program will exit now----------\n\n'
                    .format(ck_name))

            return False


    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = commons.ParamDict(meta_weights)
        inner_weights = commons.ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)
        loss = None

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
