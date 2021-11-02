import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F

from models.backbones.epi_fcr_backbones import resnet_vanilla, resnet_epi_fcr

import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp

import models.backbones.networks as networks 

from models.dm_algorithms.EPI_FCR_AGG import ModelAggregate

import numpy as np


def get_preds(logits):
    class_output = F.softmax(logits, dim=1)
    pred_task = class_output.data.max(1, keepdim=True)[1]
    return pred_task



class ModelEpiFCR(ModelAggregate):
    def __init__(self, flags, hparams, input_shape, class_balance):
        ModelAggregate.__init__(self, flags, hparams, input_shape, class_balance)
        self.iter = 0

    def configure(self):
        self.device = next(self.network.parameters()).device
        for name, para in self.ds_nn.named_parameters():
            print(name, para.size())
        for name, para in self.agg_nn.named_parameters():
            print(name, para.size())

        self.optimizer_ds_nn = optim.SGD(parameters=[{'params': self.ds_nn.parameters()}],
                                         lr=self.flags.lr,
                                         weight_decay=self.flags.weight_decay,
                                         momentum=self.flags.momentum)

        self.scheduler_ds_nn = lr_scheduler.StepLR(optimizer=self.optimizer_ds_nn, step_size=self.flags.step_size,
                                                   gamma=0.1)

        self.optimizer_agg = optim.SGD(parameters=[{'params': self.agg_nn.feature.parameters()},
                                                   {'params': self.agg_nn.classifier.parameters()}],
                                       lr=self.flags.lr,
                                       weight_decay=self.flags.weight_decay,
                                       momentum=self.flags.momentum)

        self.scheduler_agg = lr_scheduler.StepLR(optimizer=self.optimizer_agg,
                                                 step_size=self.flags.step_size, gamma=0.1)

        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def setup(self):
        torch.backends.cudnn.deterministic = self.flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        commons.fix_all_seed(self.flags.seed)

        featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        classifier = networks.Classifier(self.featurizer.n_outputs, self.flags.num_classes, self.hparams['nonlinear_classifier'])


        self.ds_nn = resnet_epi_fcr.DomainSpecificNN(backbone = featurizer, classifier= classifier, num_domains= self.num_domains).cuda()
        self.agg_nn = resnet_epi_fcr.DomainAGG(backbone = featurizer, classifier= classifier).cuda()

        self.loss_weight_epic = self.flags.loss_weight_epic
        self.loss_weight_epif = self.flags.loss_weight_epif
        self.loss_weight_epir = self.flags.loss_weight_epir

        self.best_accuracy_val = -1.0


    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.ds_nn.bn_eval()
            self.agg_nn.bn_eval()

    def update_agg_nn(self, x, y , d):
        assert(len(x) == len(y))

        taille = len(x)

        candidates = np.arange(0, len(x))

        candidates = list(candidates)
        index_val = np.random.choice(candidates, size=1)[0]
        candidates.remove(index_val)
        index_trn = np.random.choice(candidates, size=1)[0]
        assert index_trn != index_val

        self.scheduler_agg.step()

        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        epir_loss = 0.0
        epic_loss = 0.0
        epif_loss = 0.0
        for index in range(taille):
            inputs, labels, domains = x[index], y[index], d[index]
             # forward
            outputs_agg, outputs_rand, _ = self.agg_nn(x=inputs, agg_only=False)

            # loss
            agg_xent_loss += self.loss_fn(outputs_agg, labels)
            epir_loss += self.loss_fn(outputs_rand, labels) * self.loss_weight_epir
            if index == index_val:
                assert index != index_trn
                if self.iter >= self.flags.ite_train_epi_c:
                    net = self.ds_nn.features[index_trn](inputs)
                    outputs_val, _ = self.agg_nn.classifier(net)
                    epic_loss += self.loss_fn(outputs_val, labels) * self.loss_weight_epic
                if self.iter >= self.flags.ite_train_epi_f:
                    net = self.agg_nn.feature(inputs)
                    outputs_val, _ = self.ds_nn.classifiers[index_trn](net)
                    epif_loss += self.loss_fn(outputs_val, labels) * self.loss_weight_epif

        # init the grad to zeros first
        self.optimizer_agg.zero_grad()
        # backward your network
        (agg_xent_loss + epir_loss + epic_loss + epif_loss).backward()
        # optimize the parameters
        self.optimizer_agg.step()

    def update_agg_nn_warm(self, x , y, d):
        assert(len(x) == len(y))
        self.scheduler_agg.step()
        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        for index in range(len(x)):
            inputs, labels , domains = x[index], y[index], d[index]
            # forward
            outputs, _, _ = self.agg_nn(x=inputs)
            # loss
            agg_xent_loss += self.loss_fn(outputs, labels)
        # init the grad to zeros first
        self.optimizer_agg.zero_grad()
        # backward your network
        agg_xent_loss.backward()
        # optimize the parameters
        self.optimizer_agg.step()



    def update_ds_nn(self, x, y , d):
        assert(len(x) == len(y))
        self.scheduler_ds_nn.step()
        # get the inputs and labels from the data reader
        xent_loss = 0.0
        for index in range(len(x)):
            inputs, labels , domains = x[index], y[index], d[index]
            # forward
            outputs, _ = self.ds_nn(x=inputs, domain=index)
            # loss
            loss = self.loss_fn(outputs, labels)
            xent_loss += loss
        self.optimizer_ds_nn.zero_grad()
        # backward your network
        xent_loss.backward()
        # optimize the parameters
        self.optimizer_ds_nn.step()


    def predict(self, x, y , d):

        # switch on the network test mode
        self.agg_nn.eval()

        with torch.no_grad():
            logits = self.agg_nn(x, agg_only=True)

            task_predictions, _, _ = get_preds(logits).data.numpy()
            task_target = get_preds(y).data.numpy()

        self.agg_nn.train()

        return task_predictions, task_target

    # def update(self, x, y, d):
    #     flags = self.flags
    #     if self.current_epoch <= flags.loops_warm:
    #         self.update_ds_nn(x, y, d)
    #         if flags.warm_up_agg == 1 and self.current_epoch <= flags.loops_agg_warm:
    #             self.update_agg_nn_warm(x, y, d)
    #     else:
    #         self.update_ds_nn(x, y, d)
    #         self.update_agg_nn(x, y, d)

