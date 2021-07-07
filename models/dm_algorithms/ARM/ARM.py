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




class ARM(erm.ERM):
    def __init__(self, flags, backbone,  num_domains, input_shape, datasets):
        super(ARM, self).__init__(flags, backbone,  num_domains, input_shape, datasets)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.original_input_shape = input_shape
        self.input_shape = (1 + self.original_input_shape[0],) + self.original_input_shape[1:]
        self.datasets = datasets
        self.backbone = backbone
        self.flags = flags
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self):
        flags = self.flags

        self.num_devices = cuda.device_count()
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        commons.fix_all_seed(flags.seed)

        self.context_net = networks.ContextNet(self.original_input_shape)
        self.support_size = self.hparams['batch_size']
            

        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)

        self.checkpoint_vals = collections.defaultdict(lambda: [])

        self.load_state_dict(flags, self.network)

    def configure(self):

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None



    def setup_path(self):

        flags = self.flags
        train_data = self.datasets['train']
        val_data = self.datasets['val']
        test_data = self.datasets['test']


        self.dataset_sizes = {
            x: len(self.datasets[x])
            for x in ['train', 'val', 'test']
        }

        self.train_dataloaders = {}

        for source_key in train_data.keys():

            train_dataloader = torch.utils.data.DataLoader(
                train_data,
                batch_size=flags.batch_size * self.num_devices,
                shuffle=True,
                num_workers=mp.cpu_count(),
                pin_memory=True,
                worker_init_fn=commons.worker_init_fn)

            self.train_dataloaders[source_key] = train_dataloader

        self.val_dataloaders = {}

        for source_key in train_data.keys():
            val_dataloader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=flags.batch_size * self.num_devices,
                    shuffle=False,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn)

            self.val_dataloaders[source_key] = val_dataloader

        self.test_dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=flags.batch_size * self.num_devices,
                shuffle=True,
                num_workers=mp.cpu_count(),
                pin_memory=True,
                worker_init_fn=commons.worker_init_fn)

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)




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



    def adjust_learning_rate(self, flags, epoch=1, every_n=30):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = flags.init_lr * (0.1**(epoch // every_n_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_lr = lr

        
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
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)
