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

    def __init__(self, algorithm_dict, algorithm_class, flags, hparams, input_shape, datasets, class_idx,
                 checkpoint_path, class_balance):
        super(G2DMLauncher, self).__init__(algorithm_dict, algorithm_class, flags, hparams, input_shape, datasets, class_idx,
                                           checkpoint_path, class_balance)
        self.algorithm_dict = algorithm_dict
        self.algorithm_name = algorithm_class
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

    def train_workflow(self):

        flags = self.flags

        self.train_iters = {}
        for domain_key in self.train_dataloaders.keys():
            self.train_iters[domain_key] = iter(
                self.train_dataloaders[domain_key])

        self.algorithm.train()
        last_results_keys = None
        while self.current_epoch < flags.n_epochs:
            step_start_time = time()

            step_vals = {}

            inputs = []
            labels = []
            domains = []

            again = True

            while again:
                for domain_key in self.train_iters.keys():
                    again = False
                    try:
                        x_samples, y_samples, d_samples = next(
                            self.train_iters[domain_key])
                    except:
                        x_samples, y_samples, d_samples = None, None, None

                    if (x_samples is not None) and (y_samples is not None) and (d_samples is not None):
                        inputs.append(x_samples)
                        labels.append(y_samples)
                        domains.append(d_samples)
                        again = True

                inputs = torch.cat(inputs)
                labels = torch.cat(labels)
                domains = torch.cat(domains)

                inputs, labels, domains = Variable(inputs, requires_grad=False).cuda(), Variable(labels, requires_grad=False).long().cuda(), Variable( domains, requires_grad=False).long().cuda()

                iter_values = self.algorithm.update(inputs, labels, domains)

                step_vals.update(iter_values)

            self.checkpoint_values['step_time'].append(
                time() - step_start_time)
            for key, val in step_vals.items():
                self.checkpoint_values[key].append(val)

            if (self.current_epoch % self.flags.test_every == 0) or (self.current_epoch == flags.n_epochs - 1):
                results = {
                    'step': self.current_epoch,
                    'epoch': self.current_epoch / self.flags.steps_per_epoch,
                }

                for key, val in self.checkpoint_values.items():
                    results[key] = np.mean(val)

                result_dict, result_dict_per_domain = self.test_workflow()

                for domain_key in self.test_dataloaders.keys():
                    results[domain_key + '_acc'] = result_dict_per_domain['target'][domain_key]['accuracy']

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    commons.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                commons.print_row([results[key]
                                   for key in results_keys], colwidth=12)

                results.update({
                    'hparams': self.hparams,
                    'args': vars(self.flags)
                })

                epochs_path = os.path.join(
                    self.checkpoint_path, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                #self.algorithm_dict = self.algorithm.state_dict()
                self.checkpoint_values = collections.defaultdict(lambda: [])

                if self.current_epoch % self.flags.save_every:
                    self.checkpointing()



    def test_workflow(self):

        self.source_iters = {}
        for domain_key in self.val_dataloaders.keys():
            self.source_iters[domain_key] = iter(
                self.val_dataloaders[domain_key])
        self.target_iter = iter(self.test_dataloaders)

        self.test_workflow_iters = {
            'source': self.source_iters,
            'target': self.target_iter
        }

        feature_extractor = self.algorithm.feature_extractor.eval()
        task_classifier = self.algorithm.task_classifier.eval()
        disc_list = self.algorithm.disc_list

        for disc in disc_list:
            disc = disc.eval()

        result_dict = {}
        result_dict_per_domain = {}

        for source_target in self.test_workflow_iters.keys():
            result_dict[source_target], result_dict_per_domain[source_target] = self.test(source_target, feature_extractor, task_classifier, disc_list)

        return result_dict, result_dict_per_domain

    def test(self, source_target, feature_extractor, task_classifier, disc_list):

        n_total = 0
        n_correct = 0

        list_task_predictions = []
        list_task_targets = []
        list_domain_predictions = []
        list_domain_targets = []

        metrics_results_per_domain = {}

        with torch.no_grad():

            feature_extractor = feature_extractor.to(self.device)
            task_classifier = task_classifier.to(self.device)

            for disc in disc_list:
                disc = disc.to(self.device)

            for domain_key in self.test_workflow_iters[source_target].keys():

                list_task_predictions_per_domain = []
                list_task_targets_per_domain = []
                list_domain_predictions_per_domain = []
                list_domain_targets_per_domain = []

                for inputs, labels, domains in self.test_workflow_iters[source_target][domain_key]:

                    n_total += inputs.size(0)

                    inputs, labels, domains = Variable(inputs, requires_grad=False).cuda(), Variable(labels, requires_grad=False).long().cuda(), Variable( domains, requires_grad=False).long().cuda()

                    task_predictions, task_targets, domain_predictions, domain_labels = self.algorithm.predict(
                        inputs, labels, domains, feature_extractor, task_classifier, disc_list, self.device, 'source')

                    n_correct += task_predictions.eq(task_targets.data.view_as(
                        task_predictions)).cpu().sum() / task_targets.size(2)

                    list_task_predictions_per_domain.extend(
                        task_predictions.tolist())
                    list_task_targets_per_domain.extend(task_targets.tolist())
                    list_domain_predictions_per_domain.extend(
                        domain_predictions.tolist())
                    list_domain_targets_per_domain.extend(
                        domain_labels.tolist())

                metrics_results_per_domain[domain_key] = {
                    'task': {
                        'accuracy': accuracy_score(y_true=list_task_targets_per_domain, y_pred=list_task_predictions_per_domain),
                        'f1_score': f1_score(y_true=list_task_targets_per_domain, y_pred=list_task_predictions_per_domain, average='macro')
                    },
                    'domain': {
                        'accuracy': accuracy_score(y_true=list_domain_targets_per_domain, y_pred=list_domain_predictions_per_domain),
                        'f1_score': f1_score(y_true=list_domain_targets_per_domain, y_pred=list_domain_predictions_per_domain, average='macro')
                    }

                }

                list_task_predictions.extend(list_task_predictions_per_domain)
                list_task_targets.extend(list_task_targets_per_domain)
                list_domain_predictions.extend(
                    list_domain_predictions_per_domain)
                list_domain_targets.extend(list_domain_targets_per_domain)

        acc = n_correct * 1.0 / n_total

        metrics_results = {
            'task': {
                'accuracy': accuracy_score(y_true=list_task_targets, y_pred=list_task_predictions),
                'f1_score': f1_score(y_true=list_task_targets, y_pred=list_task_predictions, average='macro')
            }
        }

        if source_target == 'source':
            metrics_results['domain'] = {
                'accuracy': accuracy_score(y_true=list_domain_targets, y_pred=list_domain_predictions),
                'f1_score': f1_score(y_true=list_domain_targets, y_pred=list_domain_predictions, average='macro')
            }

        if self.writer is not None:
            self.writer.add_histogram(
                'Test/' + source_target, task_predictions, self.current_epoch)
            self.writer.add_scalar(
                'Test/' + source_target + '_accuracy', acc, self.current_epoch)

            if source_target == 'source':
                for i, disc in enumerate(disc_list):
                    self.writer.add_histogram(
                        'Test/source-D{}-pred'.format(i), domain_predictions[i], self.current_epoch)
                    self.writer.add_pr_curve('Test/ROC-D{}'.format(i), labels=list_domain_targets,
                                             predictions=list_domain_predictions, global_step=self.current_epoch)

        return metrics_results, metrics_results_per_domain

    def extract_features(self):
        raise NotImplementedError

    def vizualize_features(self):
        raise NotImplementedError



