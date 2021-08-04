import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from test import test
from utils.misc import LabelSmoothingLoss, GradualWarmupScheduler, get_values_from_batch
from torch.autograd import Variable
from models.algoritms import DefaultModel
from models.backbones.g2dm_backbones import models
import torch.utils.data
import torch.optim as optim
import utils.misc as misc
import utils.commons as commons
import models.backbones.networks as networks 
import multiprocessing as mp


def get_preds(logits):
    class_output = F.softmax(logits, dim=1)
    pred_task = class_output.data.max(1, keepdim=True)[1]
    return pred_task


class G2DM(DefaultModel):

    def __init__(self,  flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(G2DM, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.flags = flags
        self.input_shape = input_shape
        self.flags = flags
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.checkpoint_path = checkpoint_path
        self.datasets = datasets
        self.cur_epoch = 0
        self.total_iter = 0
        self.nadir_slack = flags.nadir_slack
        self.alpha = flags.alpha
        self.ablation = flags.ablation
        self.train_mode = flags.train_mode
        self.save_cp = flags.save_cp
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path, 'task' + flags.cp_name) if flags.cp_name else os.path.join(
            self.checkpoint_path, 'task_checkpoint_{}ep.pt')
        self.save_epoch_fmt_domain = os.path.join(self.checkpoint_path,
                                                  'Domain_{}' + flags.cp_name) if flags.cp_name else os.path.join(
            self.checkpoint_path, 'Domain_{}.pt')

        if flags.checkpoint_epoch is not None:
            self.load_checkpoint(flags.checkpoint_epoch)

        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)
        
        

    def setup(self):
        flags = self.flags

        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:',
              torch.backends.cudnn.deterministic)
        commons.fix_all_seed(self.flags.seed)
        #task_classifier = models.task_classifier()
        self.domain_discriminator_list = []
        for i in range(self.num_domains):
            if flags.rp_size == 4096:
                disc = models.domain_discriminator_ablation_RP(optim.SGD, flags.lr_domain, flags.momentum_domain,
                                                               flags.l2).train()
            else:
                disc = models.domain_discriminator(flags.rp_size, optim.SGD, flags.lr_domain, flags.momentum_domain,
                                                   flags.l2).train()
            self.domain_discriminator_list.append(disc)

        self.featurizer = networks.Featurizer(self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])

        # self.models_dict = {}

        # self.models_dict['feature_extractor'] = feature_extractor
        # self.models_dict['task_classifier'] = task_classifier
        # self.models_dict['domain_discriminator_list'] = domain_discriminator_list

        if flags.cuda:
            self.featurizer = self.featurizer.cuda()
            self.classifier = self.classifier.cuda()
            for k, disc in enumerate(self.domain_discriminator_list):
                self.domain_discriminator_list[k] = self.domain_discriminator_list[k].cuda()

            torch.backends.cudnn.benchmark = True


        self.cuda_mode = flags.cuda
        self.batch_size = flags.batch_size


        self.device = next(self.featurizer.parameters()).device
        # self.flow_net = None

        self.history = {'loss_task': [], 'hypervolume': [], 'loss_domain': [], 'accuracy_source': [],
                        'accuracy_target': []}

    def configure(self):
        flags = self.flags
        self.optimizer_task = optim.SGD(list(self.feature_extractor.parameters()) + list(self.task_classifier.parameters()),
                                        lr=flags.lr_task, momentum=flags.momentum_task, weight_decay=flags.l2)

        its_per_epoch = len(self.source_loader.dataset) // (self.source_loader.batch_size) + 1 if len(self.source_loader.dataset) % (
            self.source_loader.batch_size) > 0 else len(self.source_loader.dataset) // (self.source_loader.batch_size)
        patience = flags.patience * (1 + its_per_epoch)
        self.after_scheduler_task = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_task, factor=flags.factor,
                                                                               patience=patience,
                                                                               verbose=True if flags.verbose > 0 else False,
                                                                               threshold=flags.lr_threshold, min_lr=1e-7)
        self.after_scheduler_disc_list = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(disc.optimizer, factor=flags.factor, patience=patience,
                                                       verbose=True if flags.verbose > 0 else False, threshold=flags.lr_threshold,
                                                       min_lr=1e-7) for disc in self.domain_discriminator_list]

        self.scheduler_task = GradualWarmupScheduler(self.optimizer_task, total_epoch=flags.warmup_its,
                                                     after_scheduler=self.after_scheduler_task)
        self.scheduler_disc_list = [
            GradualWarmupScheduler(self.domain_discriminator_list[i].optimizer, total_epoch=flags.warmup_its,
                                   after_scheduler=sch_disc) for i, sch_disc in
            enumerate(self.after_scheduler_disc_list)]

        if flags.label_smoothing > 0.0:
            self.ce_criterion = LabelSmoothingLoss(
                flags.label_smoothing, lbl_set_size=7)
        else:
            # self.ce_criterion = torch.nn.CrossEntropyLoss()	#torch.nn.NLLLoss()#
            self.ce_criterion = F.binary_cross_entropy_with_logits
        # loss_domain_discriminator = F.binary_cross_entropy_with_logits(y_predict, curr_y_domain)
        weight = torch.tensor([2.0 / 3.0, 1.0 / 3.0]).to(self.device)
        # d_cr=torch.nn.CrossEntropyLoss(weight=weight)
        self.d_cr = torch.nn.NLLLoss(weight=weight)


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




    # self.d_cr=  F.binary_cross_entropy_with_logits()
    #### Edit####

    def adjust_learning_rate(self, optimizer, epoch=1, every_n=700, In_lr=0.01):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = In_lr * (0.1 ** (epoch // every_n_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    ######
    def train(self):
        '''
        ##### Edit ####
        # init necessary objects
        num_steps = n_epochs #* (len(train_loader.dataset) / train_loader.batch_size)
        #yd = Variable(torch.from_numpy(np.hstack([np.repeat(1, int(batch_size / 2)), np.repeat(0, int(batch_size / 2))]).reshape(50, 1)))
        j = 0
        lambd=0
        max_lambd= 1 - self.alpha

        max_lr_t = list(self.optimizer_task.param_groups)[-1]['initial_lr']

        max_lr_d = list(self.domain_discriminator_list[0].optimizer.param_groups)[-1]['initial_lr']
        #0.005
        #lr_2=0
        #lr=In_lr
        #Domain_Classifier.set_lambda(lambd)
        #print(num_steps)

        '''
        max_lr_t = list(self.optimizer_task.param_groups)[-1]['initial_lr']

        max_lr_d = list(
            self.domain_discriminator_list[0].optimizer.param_groups)[-1]['initial_lr']

        ###############

        while self.cur_epoch < self.flags.n_epochs:

            cur_loss_task = 0
            cur_hypervolume = 0
            cur_loss_total = 0

            source_iter = tqdm(enumerate(self.source_loader), disable=False)

            lr_t = max_lr_t  # / (1. + 10 * p)**0.75 #
            lr_d = max_lr_d  # / (1. + 10 * p)**0.75 #


            print('Epoch {}/{} | Alpha = {:1.3} | Lr_task = {:1.4} | Lr_dis = {:1.4} '.format(self.cur_epoch + 1,
                                                                                              self.flags.n_epochs, self.alpha,
                                                                                              lr_t, lr_d))

            for t, dict_batch in source_iter:
                # if t > 100:
                #    break
                if self.ablation == 'all' or self.ablation == 'all_multidomain':
                    cur_losses = self.train_step_ablation_all(dict_batch)
                else:
                    cur_losses = self.train_step(dict_batch)

                self.scheduler_task.step(epoch=self.total_iter, metrics=1. - self.history['accuracy_source'][
                    -1] if self.cur_epoch > 0 else np.inf)
                for sched in self.scheduler_disc_list:
                    sched.step(epoch=self.total_iter,
                               metrics=1. - self.history['accuracy_source'][-1] if self.cur_epoch > 0 else np.inf)

                cur_loss_task += cur_losses[0]
                cur_hypervolume += cur_losses[1]
                cur_loss_total += cur_losses[2]
                self.total_iter += 1

                if self.logging:
                    self.writer.add_scalar(
                        'train/task_loss', cur_losses[0], self.total_iter)
                    self.writer.add_scalar(
                        'train/hypervolume_loss', cur_losses[1], self.total_iter)
                    self.writer.add_scalar(
                        'train/total_loss', cur_losses[2], self.total_iter)

            self.history['loss_task'].append(cur_loss_task / (t + 1))
            self.history['hypervolume'].append(cur_hypervolume / (t + 1))

            print('Current task loss: {}.'.format(cur_loss_task / (t + 1)))
            print('Current hypervolume: {}.'.format(cur_hypervolume / (t + 1)))

            self.history['accuracy_source'].append(
                test(self.test_source_loader, self.feature_extractor, self.task_classifier,
                     self.domain_discriminator_list, self.device, source_target='source', epoch=self.cur_epoch,
                     tb_writer=self.writer if self.logging else None))
            self.history['accuracy_target'].append(
                test(self.target_loader, self.feature_extractor, self.task_classifier, self.domain_discriminator_list,
                     self.device, source_target='target', epoch=self.cur_epoch,
                     tb_writer=self.writer if self.logging else None))

            idx = np.argmax(self.history['accuracy_source'])

            print(
                'Valid. on SOURCE data - Current acc., best acc., best acc target, and epoch: {:0.4f}, {:0.4f}, {:0.4f}, {}'.format(
                    self.history['accuracy_source'][-1], np.max(
                        self.history['accuracy_source']),
                    self.history['accuracy_target'][idx], 1 + np.argmax(self.history['accuracy_source'])))
            print('Valid. on TARGET data - Current acc., best acc., and epoch: {:0.4f}, {:0.4f}, {}'.format(
                self.history['accuracy_target'][-1], np.max(
                    self.history['accuracy_target']),
                1 + np.argmax(self.history['accuracy_target'])))

            if self.logging:
                self.writer.add_scalar(
                    'misc/LR-task', self.optimizer_task.param_groups[0]['lr'], self.total_iter)
                for i, disc in enumerate(self.domain_discriminator_list):
                    self.writer.add_scalar('misc/LR-disc{}'.format(i), disc.optimizer.param_groups[0]['lr'],
                                           self.total_iter)

            self.cur_epoch += 1

            if self.save_cp and (self.cur_epoch % self.flags.save_every == 0 or self.history['accuracy_target'][-1] > np.max(
                    [-np.inf] + self.history['accuracy_target'][:-1])):
                self.checkpointing()

        if self.logging:
            self.writer.close()

        idx_final = np.argmax(self.history['accuracy_source'])

        return np.max(self.history['accuracy_target']), self.history['accuracy_target'][-1]

    def train_step(self, batch):
        self.feature_extractor.train()
        self.task_classifier.train()
        for disc in self.domain_discriminator_list:
            disc = disc.train()


        inputs, y_task, y_domain = get_values_from_batch(batch)

        # print(input.shape)
        t = inputs.size(2)

        if self.cuda_mode:
            # x = x.to(cuda)
            inputs = Variable(inputs.cuda())
            # y_task = y_task.to(self.device)
            y_task = Variable(y_task.cuda())

        # COMPUTING FEATURES
        # print('Computing features')
        # print(input.shape)
        features = self.feature_extractor.forward(inputs)
        # print(features.shape)
        features_ = features.detach()
        nb_features = self.batch_size * len(batch.keys())
        features_ = features_.view(nb_features, 1024)
        # print(features_.shape)

        # DOMAIN DISCRIMINATORS (First)
        for i, disc in enumerate(self.domain_discriminator_list):
            y_predict = disc.forward(features_).squeeze()

            curr_y_domain = torch.where(y_domain == i, torch.ones(
                y_domain.size(0)), torch.zeros(y_domain.size(0)))
            curr_y_domain.type_as(y_domain)
            # print(y_domain.shape,curr_y_domain.shape,y_predict.shape)
            # print(sum(curr_y_domain))
            curr_y_domain = curr_y_domain.long()
            if self.cuda_mode:
                curr_y_domain = curr_y_domain.long().to(self.device)

            # loss_domain_discriminator = F.binary_cross_entropy_with_logits(y_predict, curr_y_domain)
            # weight = torch.tensor([2.0/3.0, 1.0/3.0]).to(self.device)
            # d_cr=torch.nn.CrossEntropyLoss(weight=weight)
            # d_cr=torch.nn.NLLLoss(weight=weight)
            loss_domain_discriminator = self.d_cr(y_predict, curr_y_domain)
            # print(loss_domain_discriminator)

            if self.logging:
                self.writer.add_scalar(
                    'train/D{}_loss'.format(i), loss_domain_discriminator, self.total_iter)

            disc.optimizer.zero_grad()
            loss_domain_discriminator.backward()
            disc.optimizer.step()

        # UPDATE TASK CLASSIFIER AND FEATURE EXTRACTOR
        task_out = self.task_classifier.forward(features)

        loss_domain_disc_list = []
        loss_domain_disc_list_float = []
        for i, disc in enumerate(self.domain_discriminator_list):
            y_predict = disc.forward(features).squeeze()
            curr_y_domain = torch.where(y_domain == i, torch.zeros(
                y_domain.size(0)), torch.ones(y_domain.size(0)))

            curr_y_domain = curr_y_domain.long()
            if self.cuda_mode:
                curr_y_domain = curr_y_domain.long().to(self.device)

            # loss_domain_disc_list.append(F.binary_cross_entropy_with_logits(y_predict, curr_y_domain))
            # y_predict= y_predict.long().to(self.device)
            loss_domain_disc_list.append(self.d_cr(y_predict, curr_y_domain))

            loss_domain_disc_list_float.append(
                loss_domain_disc_list[-1].detach().item())

        if self.train_mode == 'hv':
            self.update_nadir_point(loss_domain_disc_list_float)

        hypervolume = 0
        for loss in loss_domain_disc_list:
            if self.train_mode == 'hv':
                hypervolume -= torch.log(self.nadir - loss + 1e-6)
            # hypervolume -= torch.log(loss)

            elif self.train_mode == 'avg':
                hypervolume -= loss

        # print(task_out.shape)
        task_out = torch.nn.functional.interpolate(task_out, t, mode='linear')

        task_loss = self.ce_criterion(task_out, y_task)
        loss_total = self.alpha * task_loss + \
            (1 - self.alpha) * hypervolume / len(loss_domain_disc_list)
        # loss_total = task_loss + self.alpha *hypervolume/len(loss_domain_disc_list)

        self.optimizer_task.zero_grad()
        loss_total.backward()
        self.optimizer_task.step()

        losses_return = task_loss.item(), hypervolume.item(), loss_total.item()
        return losses_return

        

    def train_step_ablation_all(self, batch):

        self.feature_extractor.train()
        self.task_classifier.train()
        for disc in self.domain_discriminator_list:
            disc = disc.train()

        # x_1, x_2, x_3, y_task_1, y_task_2, y_task_3, _, _, _ = batch

        # x = torch.cat((x_1, x_2, x_3), dim=0)
        # y_task = torch.cat((y_task_1, y_task_2, y_task_3), dim=0)

        x, y_task, _ = get_values_from_batch(batch)

        if self.cuda_mode:
            x = x.to(self.device)
            y_task = y_task.to(self.device)

        t = x.size(2)

        # COMPUTING FEATURES
        features = self.feature_extractor.forward(x)
        task_out = self.task_classifier.forward(features)
        task_out = torch.nn.functional.interpolate(task_out, t, mode='linear')

        # print(task_out.shape)
        # print(y_task.shape)
        task_loss = self.ce_criterion(task_out, y_task)
        # task_loss = torch.nn.CrossEntropyLoss()(task_out, y_task)

        self.optimizer_task.zero_grad()
        task_loss.backward()
        self.optimizer_task.step()

        losses_return = task_loss.item(), 0, task_loss.item()

        return losses_return

    def checkpointing(self):
        if self.verbose > 0:
            print(' ')
            print('Checkpointing...')

        ckpt = {'feature_extractor_state': self.feature_extractor.state_dict(),
                'task_classifier_state': self.task_classifier.state_dict(),
                'optimizer_task_state': self.optimizer_task.state_dict(),
                'scheduler_task_state': self.scheduler_task.state_dict(),
                'history': self.history,
                'cur_epoch': self.cur_epoch}
        torch.save(ckpt, self.save_epoch_fmt_task.format(self.cur_epoch))

        # print('Saving checkpoint')
        # print(ckpt)

        for i, disc in enumerate(self.domain_discriminator_list):
            ckpt = {'model_state': disc.state_dict(),
                    'optimizer_disc_state': disc.optimizer.state_dict(),
                    'scheduler_disc_state': self.scheduler_disc_list[i].state_dict()}
            torch.save(ckpt, self.save_epoch_fmt_domain.format(i + 1))

    def load_checkpoint(self, epoch):
        ckpt = self.save_epoch_fmt_task.format(epoch)

        if os.path.isfile(ckpt):

            ckpt = torch.load(ckpt)
            # Load model state
            self.feature_extractor.load_state_dict(
                ckpt['feature_extractor_state'])
            self.task_classifier.load_state_dict(ckpt['task_classifier_state'])
            # self.domain_classifier.load_state_dict(ckpt['domain_classifier_state'])
            # Load optimizer state
            self.optimizer_task.load_state_dict(ckpt['optimizer_task_state'])
            # Load scheduler state
            self.scheduler_task.load_state_dict(ckpt['scheduler_task_state'])
            # Load history
            self.history = ckpt['history']
            self.cur_epoch = ckpt['cur_epoch']

            for i, disc in enumerate(self.domain_discriminator_list):
                ckpt = torch.load(self.save_epoch_fmt_domain.format(i + 1))
                disc.load_state_dict(ckpt['model_state'])
                disc.optimizer.load_state_dict(ckpt['optimizer_disc_state'])
                self.scheduler_disc_list[i].load_state_dict(
                    ckpt['scheduler_disc_state'])

        else:
            print('No checkpoint found at: {}'.format(ckpt))

    def print_grad_norms(self, model):
        norm = 0.0
        for params in list(filter(lambda p: p.grad is not None, model.parameters())):
            norm += params.grad.norm(2).item()
        print('Sum of grads norms: {}'.format(norm))

    def update_nadir_point(self, losses_list):
        self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)
