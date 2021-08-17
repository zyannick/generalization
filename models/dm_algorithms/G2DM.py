import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.misc import LabelSmoothingLoss, GradualWarmupScheduler, get_values_from_batch
from torch.autograd import Variable
from models.algoritms import Algorithm
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


class G2DM(Algorithm):

    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(G2DM, self).__init__(flags, hparams, input_shape,
                                   datasets, checkpoint_path, class_balance)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.nadir_slack = flags.nadir_slack
        self.alpha = flags.alpha
        self.ablation = flags.ablation
        self.train_mode = flags.train_mode
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path, 'task' + flags.cp_name) if flags.cp_name else os.path.join(
            self.checkpoint_path, 'task_checkpoint_{}ep.pt')
        self.save_epoch_fmt_domain = os.path.join(self.checkpoint_path,
                                                  'Domain_{}' + flags.cp_name) if flags.cp_name else os.path.join(
            self.checkpoint_path, 'Domain_{}.pt')

        if flags.checkpoint_epoch is not None:
            self.load_checkpoint(flags.checkpoint_epoch)

        self.setup()
        self.configure()

    def setup(self):
        flags = self.flags

        torch.backends.cudnn.deterministic = flags.deterministic
        print('torch.backends.cudnn.deterministic:', torch.backends.cudnn.deterministic)
        commons.fix_all_seed(self.flags.seed)
        # task_classifier = models.task_classifier()
        self.domain_discriminator_list = []
        for i in range(self.num_domains):
            if flags.rp_size == 4096:
                disc = models.domain_discriminator_ablation_RP(optim.SGD, flags.lr_domain, flags.momentum_domain,
                                                               flags.l2).train()
            else:
                disc = models.domain_discriminator(flags.rp_size, optim.SGD, flags.lr_domain, flags.momentum_domain,
                                                   flags.l2).train()
            self.domain_discriminator_list.append(disc)

        self.featurizer = networks.Featurizer(
            self.input_shape, self.hparams, self.flags)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, flags.num_classes, self.hparams['nonlinear_classifier'])

        if flags.cuda:
            self.featurizer = self.featurizer.cuda()
            self.classifier = self.classifier.cuda()
            for k, disc in enumerate(self.domain_discriminator_list):
                self.domain_discriminator_list[k] = self.domain_discriminator_list[k].cuda(
                )

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
            self.ce_criterion = F.binary_cross_entropy_with_logits

        weight = torch.tensor([2.0 / 3.0, 1.0 / 3.0]).to(self.device)
        self.d_cr = torch.nn.NLLLoss(weight=weight)

    def update(self, x, y, d):

        if self.ablation == 'all' or self.ablation == 'all_multidomain':
            cur_losses = self.update_ablation(x, y, d)
        else:
            cur_losses = self.update_all(x, y, d)
        self.scheduler_task.step(epoch=self.total_iter, metrics=1. - self.history['accuracy_source'][-1] if self.cur_epoch > 0 else np.inf)
        for sched in self.scheduler_disc_list:
            sched.step(epoch=self.total_iter, metrics=1. - self.history['accuracy_source'][-1] if self.cur_epoch > 0 else np.inf)

        return cur_losses

    def update_all(self, x, y, d):
        self.feature_extractor.train()
        self.task_classifier.train()
        for disc in self.domain_discriminator_list:
            disc = disc.train()

        # print(input.shape)
        t = x.size(2)

        # COMPUTING FEATURES
        # print('Computing features')
        # print(input.shape)
        features = self.feature_extractor.forward(x)
        # print(features.shape)
        features_detached = features.detach()
        nb_features = self.batch_size * len(x.shape[0])
        features_detached = features_detached.view(nb_features, 1024)
        # print(features_.shape)

        # DOMAIN DISCRIMINATORS (First)
        for i, disc in enumerate(self.domain_discriminator_list):
            y_predict = disc.forward(features_detached).squeeze()

            curr_y_domain = torch.where(d == i, torch.ones(d.size(0)), torch.zeros(d.size(0)))
            curr_y_domain.type_as(d)
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
            curr_y_domain = torch.where(d == i, torch.zeros(
                d.size(0)), torch.ones(d.size(0)))

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

        task_loss = self.ce_criterion(task_out, y)
        loss_total = self.alpha * task_loss + (1 - self.alpha) * hypervolume / len(loss_domain_disc_list)
        # loss_total = task_loss + self.alpha *hypervolume/len(loss_domain_disc_list)

        self.optimizer_task.zero_grad()
        loss_total.backward()
        self.optimizer_task.step()

        losses_return = task_loss.item(), hypervolume.item(), loss_total.item()
        return losses_return

    def update_ablation(self, x, y, d):

        self.feature_extractor.train()
        self.task_classifier.train()
        for disc in self.domain_discriminator_list:
            disc = disc.train()

        t = x.size(2)

        # COMPUTING FEATURES
        features = self.feature_extractor.forward(x)
        task_out = self.task_classifier.forward(features)
        task_out = torch.nn.functional.interpolate(task_out, t, mode='linear')

        task_loss = self.ce_criterion(task_out, y)

        self.optimizer_task.zero_grad()
        task_loss.backward()
        self.optimizer_task.step()

        losses_return = task_loss.item(), 0, task_loss.item()

        return losses_return

    def predict(self, x, y, d, feature_extractor, task_classifier, disc_list, device, source_target):

        Loss = F.binary_cross_entropy_with_logits

        predictions_domain = [None for _ in disc_list]
        labels_domain = [None for _ in disc_list]

        with torch.no_grad():
            taille = x.size(2)
            features = feature_extractor.forward(x)
            # Task 
            task_out = task_classifier.forward(features)
            task_out = F.upsample(task_out, taille, mode='linear')

            task_loss = Loss(task_out, y)

            task_predictions = get_preds(task_out).data.numpy()
            task_target = get_preds(y).data.numpy()

            if source_target == 'source':
                # Domain classification
                for i, disc in enumerate(self.disc_list):
                    pred_domain = disc.forward(features).squeeze()
                    curr_y_domain = torch.where(d == i, torch.ones(d.size(0)), torch.zeros(d.size(0))).float().to(device)
                    if predictions_domain[i] is None or labels_domain[i] is None:
                        predictions_domain[i] = pred_domain.data.numpy()
                        labels_domain[i] = curr_y_domain.data.numpy()
                    else:
                        predictions_domain[i] = np.concatenate(predictions_domain[i], pred_domain.data.numpy())
                        labels_domain[i] = np.concatenate(labels_domain[i], curr_y_domain.data.numpy())

        return task_predictions, task_target, predictions_domain, labels_domain


def update_nadir_point(self, losses_list):
    self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)
