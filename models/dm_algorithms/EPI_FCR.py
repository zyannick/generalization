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

from models.dm_algorithms.EPI_FCR import ModelAggregate

import numpy as np


class ModelEpiFCR(ModelAggregate):
    def __init__(self, flags):
        ModelAggregate.__init__(self, flags)

    def configure(self, flags):

        self.device = next(self.network.parameters()).device

        for name, para in self.ds_nn.named_parameters():
            print(name, para.size())
        for name, para in self.agg_nn.named_parameters():
            print(name, para.size())

        self.optimizer_ds_nn = optim.SGD(parameters=[{'params': self.ds_nn.parameters()}],
                                         lr=flags.lr,
                                         weight_decay=flags.weight_decay,
                                         momentum=flags.momentum)

        self.scheduler_ds_nn = lr_scheduler.StepLR(optimizer=self.optimizer_ds_nn, step_size=flags.step_size,
                                                   gamma=0.1)

        self.optimizer_agg = optim.SGD(parameters=[{'params': self.agg_nn.feature.parameters()},
                                                   {'params': self.agg_nn.classifier.parameters()}],
                                       lr=flags.lr,
                                       weight_decay=flags.weight_decay,
                                       momentum=flags.momentum)

        self.scheduler_agg = lr_scheduler.StepLR(optimizer=self.optimizer_agg,
                                                 step_size=flags.step_size, gamma=0.1)

        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        commons.fix_all_seed(flags.seed)

        featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])


        self.ds_nn = resnet_epi_fcr.DomainSpecificNN(backbone = featurizer, classifier= classifier).cuda()
        self.agg_nn = resnet_epi_fcr.DomainAGG(backbone = featurizer, classifier= classifier).cuda()

        print(self.ds_nn)
        print(self.agg_nn)
        print('flags:', flags)

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        commons.write_log(flags, flags_log)

        # self.load_state_dict(flags, self.ds_nn.feature1)
        # self.load_state_dict(flags, self.ds_nn.feature2)
        # self.load_state_dict(flags, self.ds_nn.feature3)
        # self.load_state_dict(flags, self.agg_nn.feature)

        self.configure(flags)

        self.loss_weight_epic = flags.loss_weight_epic
        self.loss_weight_epif = flags.loss_weight_epif
        self.loss_weight_epir = flags.loss_weight_epir

        self.best_accuracy_val = -1.0

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

        self.candidates = np.arange(0, len(self.train_dataloaders))

    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.ds_nn.bn_eval()
            self.agg_nn.bn_eval()

    def train_agg_nn(self, ite, flags):
        candidates = list(self.candidates)
        index_val = np.random.choice(candidates, size=1)[0]
        candidates.remove(index_val)
        index_trn = np.random.choice(candidates, size=1)[0]
        assert index_trn != index_val

        self.scheduler_agg.step(ite)

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

        if ite % 500 == 0 or ite < flags.loops_warm + 500:
            print(
                'ite:', ite,
                'agg_xent_loss:', agg_xent_loss.item(),
                'epir_loss:', epir_loss.item(),
                'epic_loss:', epic_loss.item() if type(epic_loss) is not float else epic_loss,
                'epif_loss:', epif_loss.item() if type(epif_loss) is not float else epif_loss,
                'lr:', self.scheduler_agg.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'agg_xent_loss.txt')
        commons.write_log(str(agg_xent_loss.item()), flags_log)
        flags_log = os.path.join(flags.logs, 'epir_loss.txt')
        commons.write_log(str(epir_loss.item()), flags_log)
        if ite >= flags.ite_train_epi_c:
            flags_log = os.path.join(flags.logs, 'epic_loss.txt')
            commons.write_log(str(epic_loss.item()), flags_log)
        if ite >= flags.ite_train_epi_f:
            flags_log = os.path.join(flags.logs, 'epif_loss.txt')
            commons.write_log(str(epif_loss.item()), flags_log)

    def train_agg_nn_warm(self, ite, flags):

        self.scheduler_agg.step(ite)

        # get the inputs and labels from the data reader
        agg_xent_loss = 0.0
        for source_key, source_loader in self.train_dataloaders.items():
            inputs_train, labels_train = source_loader

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs_train, requires_grad=False).cuda(),   Variable(labels_train, requires_grad=False).long().cuda()
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

        if ite % 500 == 0 or ite < flags.loops_warm + 500:
            print(
                'ite:', ite,
                'agg_xent_loss:', agg_xent_loss.item(),
                'lr:', self.scheduler_agg.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'agg_xent_loss.txt')
        commons.write_log(str(agg_xent_loss.item()), flags_log)

    def train_ds_nn(self, ite, flags):

        self.scheduler_ds_nn.step(ite)

        # get the inputs and labels from the data reader
        xent_loss = 0.0

        list_sources_keys = list(self.train_dataloaders.keys())

        for index in range(len(list_sources_keys)):
            source_key = list_sources_keys[index]

            source_loader = self.train_dataloaders[source_key]

            inputs_train, labels_train = source_loader

            # wrap the inputs and labels in Variable
            inputs, labels = Variable(inputs_train, requires_grad=False).cuda(), \
                             Variable(labels_train, requires_grad=False).long().cuda()

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

        if ite % 500 == 0 or ite < 500:
            print(
                'ite:', ite,
                'xent_loss:', xent_loss.item(),
                'lr:', self.scheduler_ds_nn.get_lr()[0])

        flags_log = os.path.join(flags.logs, 'ds_nn_loss_log.txt')
        commons.write_log(str(xent_loss.item()), flags_log)

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

    def test(self, flags, ite, log_prefix, log_dir='logs/', val_loader=None):

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

    def train(self):

        flags = self.flags
        self.ds_nn.train()
        self.agg_nn.train()
        self.bn_process(flags)

        for self.cur_epoch in range(flags.n_epochs):

            if self.cur_epoch <= flags.loops_warm:

                self.train_ds_nn(self.cur_epoch, flags)

                if flags.warm_up_agg == 1 and self.cur_epoch <= flags.loops_agg_warm:
                    self.train_agg_nn_warm(self.cur_epoch, flags)

                if self.cur_epoch % flags.test_every == 0 and self.cur_epoch is not 0 and flags.warm_up_agg != 1:
                    self.test_workflow(self.batImageGenVals, flags, self.cur_epoch, prefix='')

            else:

                self.train_ds_nn(self.cur_epoch, flags)
                self.train_agg_nn(self.cur_epoch, flags)

                if self.cur_epoch % flags.test_every == 0 and self.cur_epoch is not 0:
                    self.test_workflow(self.batImageGenVals, flags, self.cur_epoch, prefix='')
