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

    # def load_checkpoint(self, epoch):
    #     ckpt = self.save_epoch_fmt_task.format(epoch)
    #     ck_name = ckpt
    #     if os.path.isfile(ckpt):
    #         ckpt = torch.load(ckpt)
    #         # Load model state
    #         self.featurizer.load_state_dict(ckpt['featurizer'])
    #         self.classifier.load_state_dict(ckpt['classifier'])
    #         # Load history
    #         self.history = ckpt['history']
    #         self.cur_epoch = ckpt['cur_epoch']
    #         self.current_lr = ckpt['lr']
    #         print('Checkpoint number {} loaded  ---> {}'.format(
    #             epoch, ck_name))
    #         return True
    #     else:
    #         print(colored('No checkpoint found at: {}'.format(ckpt), 'red'))
    #         if self.flags.phase != 'train':
    #             raise ValueError(
    #                 '----------Unable to load checkpoint  {}. The program will exit now----------\n\n'
    #                 .format(ck_name))
    #         return False


    # def training(self):

    #     flags = self.flags

    #     train_minibatches_iterator = zip(*self.train_dataloaders)
    #     self.network.train()
    #     self.network.bn_eval()
    #     self.best_accuracy_val = -1
    #     while self.cur_epoch < flags.n_epochs:
    #         step_start_time = time.time()
    #         minibatches_device = [(x.cuda(), y.cuda())
    #             for x,y in next(train_minibatches_iterator)]
    #         step_vals = self.update(minibatches_device)
    #         self.checkpoint_vals['step_time'].append(time.time() - step_start_time)
    #         for key, val in step_vals.items():
    #             self.checkpoint_vals[key].append(val)
            
    #         if (self.cur_epoch % self.flags.test_every == 0) or (self.cur_epoch == flags.n_epochs - 1):
    #             results = {
    #                 'step': self.cur_epoch,
    #                 'epoch': self.cur_epoch / self.flags.steps_per_epoch,
    #             }

    #             for key, val in checkpoint_vals.items():
    #                 results[key] = np.mean(val)

    #             for domain_key, loader in self.val_dataloader:
    #                 acc = self.testing(loader, domain_key)
    #                 results[domain_key+'_acc'] = acc

    #             results_keys = sorted(results.keys())
    #             if results_keys != last_results_keys:
    #                 misc.print_row(results_keys, colwidth=12)
    #                 last_results_keys = results_keys
    #             misc.print_row([results[key] for key in results_keys], colwidth=12)

    #             results.update({
    #                 'hparams': self.hparams,
    #                 'args': vars(self.flags)
    #             })

    #             epochs_path = os.path.join(args.output_dir, 'results.jsonl')
    #             with open(epochs_path, 'a') as f:
    #                 f.write(json.dumps(results, sort_keys=True) + "\n")

    #             algorithm_dict = algorithm.state_dict()
    #             start_step = step + 1
    #             checkpoint_vals = collections.defaultdict(lambda: [])

    #             if args.save_model_every_checkpoint:
    #                 self.save_checkpoint(f'model_step{step}.pkl')

    # def testing(self, domain_key, loader):
    #     correct = 0
    #     total = 0
    #     weights_offset = 0

    #     self.network.eval()
    #     with torch.no_grad():
    #         for x, y in loader:
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             p = self.network.predict(x)

    #     self.network.train()

    #     return p
            

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
