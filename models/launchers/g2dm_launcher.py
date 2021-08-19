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
from models.algoritms import  *
from tqdm import tqdm
from datetime import datetime
from default_launcher import DefaultLauncher
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


def get_preds(logits):
    class_output = F.softmax(logits, dim=1)
    pred_task = class_output.data.max(1, keepdim=True)[1]
    return pred_task


class G2DMLauncher(DefaultLauncher):

    def __init__(self, algorithm_dict, algorithm_name, flags, hparams, input_shape, datasets, class_idx,
                 checkpoint_path, class_balance):
        super(G2DMLauncher, self).__init__(algorithm_dict, algorithm_name, flags, hparams, input_shape, datasets, class_idx,
                 checkpoint_path, class_balance)
        self.algorithm_dict = algorithm_dict
        self.algorithm_name = algorithm_name
        self.class_balance = class_balance
        self.input_shape = input_shape
        self.class_idxs = class_idx
        self.flags = flags
        self.logging = flags.logging
        self.launch_mode = flags.launch_mode
        self.num_classes = flags.num_classes
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.datasets = datasets
        self.checkpoint_path = checkpoint_path
        self.setup_path()
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')
        self.save_epoch_fmt_domain = os.path.join(self.checkpoint_path,
                                                'domain_{}ep.pt')
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.algorithm_class = get_algorithm_class(self.algorithm_name)
        self.algorithm = self.algorithm_class(flags, hparams, input_shape, datasets, class_idx, checkpoint_path,
                                              class_balance)
        # if self.algorithm_dict is not None:
        #     self.algorithm.load_state_dict(algorithm_dict)
        # self.algorithm.to(self.device)
        self.checkpoint_values = collections.defaultdict(lambda: [])
        self.current_epoch = 0
        if self.logging:
            self.writer = SummaryWriter(log_dir=os.path.join(
                'runs', self.launch_mode + '_' + str(datetime.now())))

    def checkpointing(self):
        if self.flags.verbose > 0:
            print(' ')
            print('Checkpointing...')

        ckpt = {'feature_extractor_state': self.algorithm.feature_extractor.state_dict(),
                'task_classifier_state': self.algorithm.task_classifier.state_dict(),
                'optimizer_task_state': self.algorithm.optimizer_task.state_dict(),
                'scheduler_task_state': self.algorithm.scheduler_task.state_dict(),
                'history': self.algorithm.history,
                'cur_epoch': self.cur_epoch}
        torch.save(ckpt, self.save_epoch_fmt_task.format(self.cur_epoch))

        # print('Saving checkpoint')
        # print(ckpt)

        for i, disc in enumerate(self.algorithm.domain_discriminator_list):
            ckpt = {'model_state': disc.state_dict(),
                    'optimizer_disc_state': disc.optimizer.state_dict(),
                    'scheduler_disc_state': self.algorithm.scheduler_disc_list[i].state_dict()}
            torch.save(ckpt, self.save_epoch_fmt_domain.format(i + 1))

    def load_checkpoint(self, epoch):
        ckpt = self.save_epoch_fmt_task.format(epoch)

        if os.path.isfile(ckpt):

            ckpt = torch.load(ckpt)
            # Load model state
            self.algorithm.feature_extractor.load_state_dict(
                ckpt['feature_extractor_state'])
            self.algorithm.task_classifier.load_state_dict(
                ckpt['task_classifier_state'])
            # self.domain_classifier.load_state_dict(ckpt['domain_classifier_state'])
            # Load optimizer state
            self.algorithm.optimizer_task.load_state_dict(
                ckpt['optimizer_task_state'])
            # Load scheduler state
            self.algorithm.scheduler_task.load_state_dict(
                ckpt['scheduler_task_state'])
            # Load history
            self.algorithm.history = ckpt['history']
            self.cur_epoch = ckpt['cur_epoch']

            for i, disc in enumerate(self.algorithm.domain_discriminator_list):
                ckpt = torch.load(self.save_epoch_fmt_domain.format(i + 1))
                disc.load_state_dict(ckpt['model_state'])
                disc.optimizer.load_state_dict(ckpt['optimizer_disc_state'])
                self.algorithm.scheduler_disc_list[i].load_state_dict(
                    ckpt['scheduler_disc_state'])

        else:
            print('No checkpoint found at: {}'.format(ckpt))





