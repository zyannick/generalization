import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch
import torch.nn.functional as F
import torch.cuda as cuda

import utils.commons as commons
import multiprocessing as mp
import models.backbones.networks as networks 
from termcolor import colored
import time
import collections
from models.dm_algorithms import ERM


class ARM(ERM.ERM):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.original_input_shape = input_shape
        self.input_shape = (1 + self.original_input_shape[0],) + self.original_input_shape[1:]
        super(ARM, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        self.datasets = datasets
        # self.backbone = gears['backbone']
        # self.discriminator = gears['disc']
        # self.classifier = gears['classifier']
        self.flags = flags
        self.setup()
        self.setup_path()
        self.configure()

    def setup(self):
        flags = self.flags
        self.num_devices = cuda.device_count()
        commons.fix_all_seed(flags.seed)
        self.context_net = networks.ContextNet(self.original_input_shape)
        self.support_size = self.hparams['batch_size']
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)


    # def load_checkpoint(self, epoch):
    #     ckpt = self.save_epoch_fmt_task.format(epoch)
    #     ck_name = ckpt
    #     if os.path.isfile(ckpt):
    #         ckpt = torch.load(ckpt)
    #         # Load model state
    #         self.context_net.load_state_dict(ckpt['context_net'])
    #         # Load history
    #         self.history = ckpt['history']
    #         self.cur_epoch = ckpt['cur_epoch']
    #         self.current_lr = ckpt['lr']
    #
    #         print('Checkpoint number {} loaded  ---> {}'.format(
    #             epoch, ck_name))
    #         return True
    #     else:
    #         print(colored('No checkpoint found at: {}'.format(ckpt), 'red'))
    #         if self.flags.phase != 'train':
    #             raise ValueError(
    #                 '----------Unable to load checkpoint  {}. The program will exit now----------\n\n'
    #                 .format(ck_name))
    #
    #         return False


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
