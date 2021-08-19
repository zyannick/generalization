import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch
import torch
import torch.nn as nn

from models.backbones.epi_fcr_backbones import resnet_vanilla, resnet_epi_fcr

from models.algoritms import Algorithm
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp
import models.backbones.networks as networks 



import numpy as np




class ModelAggregate(Algorithm):
    def __init__(self, flags, hparams, input_shape, class_balance):
        super(ModelAggregate, self).__init__(flags, hparams, input_shape, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.setup()
        self.configure()

    def setup(self):
        flags = self.flags
        commons.fix_all_seed(flags.seed)
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential()
        self.network.add_module(self.featurizer)
        self.network.add_module(self.classifier)
        self.network = self.network.cuda()
        print(self.network)
        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)
        self.load_state_dict(flags, self.network)

    @property
    def factory(self):
        return {'network': self.network}


    def configure(self):
        for name, para in self.network.named_parameters():
            print(name, para.size())
        self.device = next(self.network.parameters()).device
        total_params = 0
        named_params_to_update = {}
        for name, param in self.network.named_parameters():
            total_params += 1
            if param.requires_grad:
                named_params_to_update[name] = param
        print("Params to learn:")
        if len(named_params_to_update) == total_params:
            print("\tfull network")
        else:
            for name in named_params_to_update:
                print("\t{}".format(name))
        self.optimizer = optim.SGD(parameters=self.network.parameters(),
                             lr=self.flags.lr,
                             weight_decay=self.flags.weight_decay,
                             momentum=self.flags.momentum)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.flags.step_size, gamma=0.1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def adjust_learning_rate(self, flags, epoch=1, every_n=30):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = flags.init_lr * (0.1**(epoch // every_n_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr
   
    def update(self, x, y, d):
        self.network.train()
        if self.flags.input_mode == 'idt':
            self.network.bn_eval()
        self.best_accuracy_val = -1

        if self.flags.scheduler is None:
            self.adjust_learning_rate()
 
        # forward with the adapted parameters
        outputs = self.network(x)
        # loss
        loss = self.loss_fn(outputs, y)

        # init the grad to zeros first
        self.optimizer.zero_grad()
        # backward your network
        loss.backward()
        # optimize the parameters
        self.optimizer.step()

        if self.current_epoch % self.flags.test_every == 0 and self.current_epoch is not 0:
            self.test_workflow(self.val_dataloaders, self.flags, self.current_epoch)



    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.network.bn_eval()

    def predict(self, x, y, d):
        self.network.eval()
        with torch.no_grad():
            return self.network(x)

