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

from models.algoritms import Algorithm
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp
import models.backbones.networks as networks
from termcolor import colored, COLORS
import time
import collections

import numpy as np


class AbstractDANN(Algorithm):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance, conditional):
        super(AbstractDANN, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

    def setup(self):
        flags = self.flags

        commons.fix_all_seed(flags.seed)

        # Algorithms
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes,
                                              self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, flags.num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(flags.num_classes, self.featurizer.n_outputs)

        self.featurizer = self.featurizer.cuda()
        self.classifier = self.classifier.cuda()
        self.discriminator = self.discriminator.cuda()
        self.class_embeddings = self.class_embeddings.cuda()

        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)

        self.checkpoint_vals = collections.defaultdict(lambda: [])

    def configure(self):

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, x, y, d):
        device = "cuda" if x.is_cuda else "cpu"
        self.update_count += 1
        all_z = self.featurizer(x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x_.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x_, y_) in enumerate([(x, y)])
        ])

        if self.class_balance:
            y_counts = F.one_hot(y).sum(dim=0)
            weights = 1. / (y_counts[y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}
