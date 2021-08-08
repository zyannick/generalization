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

from models.algoritms import DefaultModel
import utils.commons as commons
from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
import torch.optim as optim
import multiprocessing as mp
import models.backbones.networks as networks 



import numpy as np




class ModelAggregate(DefaultModel):
    def __init__(self, flags, gears, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(ModelAggregate, self).__init__(flags, gears, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.datasets = datasets
        self.setup()
        self.setup_path()
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



    def load_state_dict(self, flags, nn):

        if flags.state_dict:

            try:
                tmp = torch.load(flags.state_dict)
                if 'state' in tmp.keys():
                    pretrained_dict = tmp['state']
                else:
                    pretrained_dict = tmp
            except:
                pretrained_dict = model_zoo.load_url(flags.state_dict)

            model_dict = nn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

            print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
            print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            nn.load_state_dict(model_dict)

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

        
    def training(self):
        self.network.train()
        if self.flags.input_mode == 'idt':
            self.network.bn_eval()
        self.best_accuracy_val = -1

        for self.current_epoch in range(self.flags.n_epochs):

            self.scheduler.step(epoch=self.current_epoch)

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

            if self.current_epoch < 500 or self.current_epoch % 500 == 0:
                print(
                    'ite:', self.current_epoch, 'total loss:', total_loss.cpu().item(), 'lr:',
                    self.scheduler.get_lr()[0])

            flags_log = os.path.join(self.flags.logs, 'loss_log.txt')
            commons.write_log(
                str(total_loss.item()),
                flags_log)

            if self.current_epoch % self.flags.test_every == 0 and self.current_epoch is not 0:
                self.test_workflow(self.val_dataloaders, self.flags, self.current_epoch)

    def test_workflow(self, validation_loaders, flags, ite):

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

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def bn_process(self, flags):
        if flags.bn_eval == 1:
            self.network.bn_eval()

    def test(self, flags, ite, log_prefix, log_dir='logs/', val_loader=None):

        # switch on the network test mode
        self.network.eval()

        if val_loader is None:
            return

        test_image_preds = []
        ground_truth = []

        for inputs_test, labels_test in val_loader:

            inputs = Variable(inputs_test, requires_grad=False).cuda()
            tuples = self.network(inputs)

            predictions = tuples[-1]['Predictions']
            predictions = predictions.cpu().data.numpy()
            test_image_preds.append(predictions)
            ground_truth.append(labels_test)

        accuracy = commons.compute_accuracy(predictions=test_image_preds, labels=ground_truth)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()
        self.bn_process(flags)

        return accuracy

