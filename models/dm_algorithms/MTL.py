from models import backbones
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
from models.dm_algorithms.ERM import ERM
import utils.commons as commons
from .AbstractMMD import  AbstractMMD
from ..algoritms import DefaultModel
import models.backbones.networks as networks
from termcolor import colored, COLORS


class MTL(DefaultModel):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(MTL, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        self.datasets = datasets
        # self.backbone = gears['backbone']
        # self.discriminator = gears['disc']
        # self.classifier = gears['classifier']
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
        self.featurizer = networks.Featurizer(self.input_shape, self.flags, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            self.num_classes,
            self.hparams['nonlinear_classifier'])

    def configure(self):

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + \
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(self.num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def load_checkpoint(self, epoch):

        ckpt = self.save_epoch_fmt_task.format(epoch)

        ck_name = ckpt

        if os.path.isfile(ckpt):

            ckpt = torch.load(ckpt)
            # Load model state
            self.featurizer.load_state_dict(ckpt['featurizer'])
            self.classifier.load_state_dict(ckpt['classifier'])
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


    def update(self, minibatches, unlabeled=None):
            loss = 0
            for env, (x, y) in enumerate(minibatches):
                loss += F.cross_entropy(self.predict(x, env), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))
