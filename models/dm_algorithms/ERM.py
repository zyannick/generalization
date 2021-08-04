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




class ERM(DefaultModel):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(ERM, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.network = self.network.cuda()
        if self.num_devices > 1:
            self.network = nn.DataParallel(self.network)
            

        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)

        self.checkpoint_vals = collections.defaultdict(lambda: [])

    def configure(self):

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

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


    def training(self):

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
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
