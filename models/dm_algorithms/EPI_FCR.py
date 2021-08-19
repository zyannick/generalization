import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch

from models.backbones.epi_fcr_backbones import resnet_vanilla, resnet_epi_fcr

from models.algoritms import DefaultModel
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp

import models.backbones.networks as networks 

from models.dm_algorithms.EPI_FCR_AGG import ModelAggregate

import numpy as np


class ModelEpiFCR(ModelAggregate):
    def __init__(self, flags):
        ModelAggregate.__init__(self, flags)

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

    def update_agg_nn(self, x, y , d, flags):
        assert(len(x) == len(y))

        candidates = np.arange(0, len(x))

        candidates = list(candidates)
        index_val = np.random.choice(candidates, size=1)[0]
        candidates.remove(index_val)
        index_trn = np.random.choice(candidates, size=1)[0]
        assert index_trn != index_val

        self.scheduler_agg.step()

        list_sources_keys = list(self.train_dataloaders.keys())

        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        epir_loss = 0.0
        epic_loss = 0.0
        epif_loss = 0.0
        for index in range(len(list_sources_keys)):
            source_key = list_sources_keys[index]
            source_loader = self.train_dataloaders[source_key]
            inputs_train, labels_train = source_loader
            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs_train, requires_grad=False).cuda(),  Variable(labels_train, requires_grad=False).long().cuda()
            # forward
            outputs_agg, outputs_rand, _ = self.agg_nn(x=inputs, agg_only=False)

            # loss
            agg_xent_loss += self.loss_fn(outputs_agg, labels)
            epir_loss += self.loss_fn(outputs_rand, labels) * self.loss_weight_epir
            if index == index_val:
                assert index != index_trn
                if ite >= flags.ite_train_epi_c:
                    net = self.ds_nn.features[index_trn](inputs)
                    outputs_val, _ = self.agg_nn.classifier(net)
                    epic_loss += self.loss_fn(outputs_val, labels) * self.loss_weight_epic
                if ite >= flags.ite_train_epi_f:
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

    def test_workflow(self, validation_loaders, flags, ite, prefix=''):

        accuracies = []
        for count, (_, val_loader) in enumerate(validation_loaders):
            accuracy_val = self.test(val_loader=val_loader, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            acc_test = self.test(val_loader=self.test_dataloader, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, '{}best_agg.tar'.format(prefix))
            torch.save({'ite': ite, 'state': self.agg_nn.state_dict()}, outfile)

    def predict(self, x, y , d):

        # switch on the network test mode
        self.agg_nn.eval()

        test_image_preds = []
        ground_truth = []

        for inputs_test, labels_test in val_loader:
            inputs = Variable(inputs_test, requires_grad=False).cuda()
            tuples = self.agg_nn(inputs, agg_only=True)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()
            test_image_preds.append(predictions)
            ground_truth.append(labels_test)

        accuracy = commons.compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.agg_nn.train()
        self.bn_process(flags)

        return accuracy

    def update(self, x, y, d):
        flags = self.flags
        if self.current_epoch <= flags.loops_warm:
            self.update_ds_nn()
            if flags.warm_up_agg == 1 and self.current_epoch <= flags.loops_agg_warm:
                self.update_agg_nn_warm()
        else:
            self.update_ds_nn()
            self.update_agg_nn()

