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




class AbstractDANN(DefaultModel):
    def __init__(self, flags, backbone,  num_domains, input_shape, datasets, class_balance, conditional):
        super(AbstractDANN, self).__init__(flags, backbone,  num_domains, input_shape, datasets, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

    def setup(self):
        flags = self.flags

        self.num_devices = cuda.device_count()

        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        commons.fix_all_seed(flags.seed)

        # Algorithms
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, flags.num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(flags.num_classes, self.featurizer.n_outputs)

        self.featurizer = self.featurizer.cuda()
        if self.num_devices > 1:
            self.featurizer = nn.DataParallel(self.featurizer)
            
        self.classifier = self.classifier.cuda()
        self.discriminator = self.discriminator.cuda()
        self.class_embeddings = self.class_embeddings.cuda()

        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)

        self.checkpoint_vals = collections.defaultdict(lambda: [])

        self.load_state_dict(flags, self.network)

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
            if self.flags.balance_sampler and len(self.class_idxs) == 2:
                batch_size = flags.batch_size * self.num_devices
                n_batches = self.dataset_sizes['train'] // batch_size
                batch_sampler = SamplerFactory().get(
                        class_idxs=self.class_idxs,
                        batch_size=batch_size,
                        n_batches=n_batches,
                        alpha=self.flags.alpha_sampler,
                        kind='fixed'
                    )     
                train_dataloader = torch.utils.data.DataLoader(
                    train_data,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn, 
                    batch_sampler=batch_sampler)
            else:
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
            self.featurizer.load_state_dict(ckpt['featurizer'])
            self.classifier.load_state_dict(ckpt['classifier'])
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.class_embeddings.load_state_dict(ckpt['class_embeddings'])
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

            # get the inputs and labels from the data reader
            total_loss = 0.0
            for source_key, source_loader in self.train_dataloaders.items():

                inputs_train, labels_train = source_loader

                # wrap the inputs and labels in Variable
                inputs, labels = Variable(inputs_train, requires_grad=False).cuda(), \
                                 Variable(labels_train, requires_grad=False).long().cuda()

                # forward with the adapted parameters
                outputs, _ = self.network(x=inputs)

                # loss
                loss = self.loss_fn(outputs, labels)

                total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            if self.cur_epoch < 500 or self.cur_epoch % 500 == 0:
                print(
                    'ite:', self.cur_epoch, 'total loss:', total_loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            commons.write_log(
                str(total_loss.item()),
                flags_log)

            if self.cur_epoch % flags.test_every == 0 and self.cur_epoch is not 0:
                self.test_workflow(self.val_dataloaders, flags, self.cur_epoch)

            self.cur_epoch += 1

            

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}
