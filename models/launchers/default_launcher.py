# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
# from torch.autograd import Variable
# import collections
# import utils.commons as commons
# import multiprocessing as mp
# import os
# import torch.cuda as cuda
# from time import time
# import json
# import numpy as np
# from data_helpers.pytorch_balanced_sampler.sampler import SamplerFactory
from models.algoritms import  *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class DefaultLauncher(object):

    def __init__(self, algorithm_dict, algorithm_name, flags, hparams, input_shape, datasets, class_idx,
                 checkpoint_path, class_balance):
        super(DefaultLauncher, self).__init__()
        self.train_dataloaders = {}
        self.algorithm_dict = algorithm_dict
        self.algorithm_name = algorithm_name
        self.class_balance = class_balance
        self.input_shape = input_shape
        self.class_idxs = class_idx
        self.flags = flags
        self.num_classes = flags.num_classes
        self.num_domains = flags.num_domains
        self.hparams = hparams
        self.datasets = datasets
        self.checkpoint_path = checkpoint_path
        self.logging = flags.logging
        self.launch_mode = flags.launch_mode
        self.setup_path()
        self.save_epoch_model_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.algorithm_class = get_algorithm_class(self.algorithm_name)
        self.algorithm = self.algorithm_class(flags, hparams, input_shape, datasets, class_idx, checkpoint_path,
                                              class_balance)
        if self.algorithm_dict is not None:
            self.algorithm.load_state_dict(algorithm_dict)
        self.algorithm.to(self.device)
        self.checkpoint_values = collections.defaultdict(lambda: [])
        self.current_epoch = 0
        if self.logging:
            self.writer = SummaryWriter(log_dir=os.path.join(
                'runs', self.launch_mode + '_' + str(datetime.now())))

    def setup_path(self):
        assert (self.flags.class_balanced != self.flags.balance_sampler)
        flags = self.flags

        train_data = self.datasets['train']

        val_data = self.datasets['val']

        test_data = self.datasets['test']

        self.dataset_sizes = {
            x: len(self.datasets[x])
            for x in ['train', 'val', 'test']
        }

        self.train_minibatches_iterator = {}

        self.train_weights = {}
        self.train_sampler = {}

        self.val_weights = {}
        self.val_sampler = {}

        if self.flags.class_balanced:
            for train_key in train_data.keys():
                self.train_weights[train_key] = commons.make_weights_for_balanced_classes(train_data[train_key])
            for val_key in val_data.keys():
                self.val_weights[val_key] = commons.make_weights_for_balanced_classes(val_data[val_key])
        else:
            for train_key in train_data.keys():
                self.train_weights[train_key] = None
            for val_key in val_data.keys():
                self.val_weights[val_key] = None

        for train_key in train_data.keys():
            if self.flags.balance_sampler:
                batch_size = flags.batch_size
                n_batches = self.dataset_sizes['train'] // batch_size
                train_list_idx = []
                for _, class_key in enumerate(list(self.class_idxs['train'].keys())):
                    train_list_idx.append(self.class_idxs['train'][class_key])
                batch_sampler = SamplerFactory().get(
                    class_idxs=train_list_idx,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    alpha=self.flags.alpha_sampler,
                    kind='fixed'
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_data[train_key],
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    batch_sampler=batch_sampler,
                    worker_init_fn=commons.worker_init_fn)
            elif self.flags.class_balanced:
                train_dataloader = torch.utils.data.DataLoader(
                    train_data[train_key],
                    batch_size=flags.batch_size,
                    weights=self.train_weights[train_key],
                    shuffle=True,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn)
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    train_data[train_key],
                    batch_size=flags.batch_size,
                    shuffle=True,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn)

            self.train_dataloaders[train_key] = train_dataloader

        self.val_dataloaders = {}

        for val_key in val_data.keys():
            if self.flags.balance_sampler:
                batch_size = flags.batch_size
                n_batches = self.dataset_sizes['val'] // batch_size
                val_list_idx = []
                for _, class_key in enumerate(list(self.class_idxs['val'].keys())):
                    val_list_idx.append(self.class_idxs['val'][class_key])
                batch_sampler = SamplerFactory().get(
                    class_idxs=val_list_idx,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    alpha=self.flags.alpha_sampler,
                    kind='fixed'
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_data[val_key],
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    batch_sampler=batch_sampler,
                    worker_init_fn=commons.worker_init_fn)
            elif self.flags.class_balanced:
                val_dataloader = torch.utils.data.DataLoader(
                    val_data[val_key],
                    batch_size=flags.batch_size,
                    weights=self.val_weights[val_key],
                    shuffle=True,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn)
            else:
                val_dataloader = torch.utils.data.DataLoader(
                    val_data[val_key],
                    batch_size=flags.batch_size,
                    shuffle=True,
                    num_workers=mp.cpu_count(),
                    pin_memory=True,
                    worker_init_fn=commons.worker_init_fn)

            self.val_dataloaders[val_key] = val_dataloader

        self.test_dataloaders = {}

        for test_key in test_data.keys():
            self.test_dataloaders[test_key] = torch.utils.data.DataLoader(
                test_data[test_key],
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=mp.cpu_count(),
                pin_memory=True,
                worker_init_fn=commons.worker_init_fn)

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

    

    def checkpointing(self):
        if self.flags.skip_model_save:
            return
        save_dict = {
            'args': vars(self.flags),
            'model_num_classes': self.num_classes,
            'model_num_domains': self.num_domains,
            'model_hparams': self.hparams,
            'model_dict': self.algorithm.cpu().state_dict(),
            'history': self.history
        }
        torch.save(save_dict, os.path.join(self.checkpoint_path, f'model_{self.current_epoch}.pt'))

    def load_checkpoint(self, epoch):
        ckpt = self.save_epoch_model_task.format(epoch)
        ck_name = ckpt
        if os.path.isfile(ckpt):
            ckpt = torch.load(ckpt)
            # Load model state
            self.algorithm.load_state_dict(ckpt['model_dict'])
            # Load history
            self.history = ckpt['history']
            # Load num classes
            self.num_classes = ckpt['model_num_classes']
            # Load num domains
            self.model_num_domains = ckpt['model_num_domains']
            # Load hparams
            self.hparams = ckpt['model_hparams']
            # Load flags
            self.flags = ckpt['args']

            print('Checkpoint number {} loaded  ---> {}'.format(epoch, ck_name))

            return True

        return False

    def train_workflow(self):

        flags = self.flags

        self.train_iters = {}
        for domain_key in self.train_dataloaders.keys():
            self.train_iters[domain_key] = iter(self.train_dataloaders[domain_key])

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
                        x_samples, y_samples, d_samples = next(self.train_iters[domain_key])
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

                inputs = inputs.cuda()
                labels = labels.cuda()
                domains = domains.cuda()

                iter_values = self.algorithm.update(inputs, labels, domains)

                step_vals.update(iter_values)

            self.checkpoint_values['step_time'].append(time() - step_start_time)
            for key, val in step_vals.items():
                self.checkpoint_values[key].append(val)

            if (self.current_epoch % self.flags.test_every == 0) or (self.current_epoch == flags.n_epochs - 1):
                results = {
                    'step': self.current_epoch,
                    'epoch': self.current_epoch / self.flags.steps_per_epoch,
                }

                for key, val in self.checkpoint_values.items():
                    results[key] = np.mean(val)

                for domain_key, loader in self.val_dataloaders:
                    acc = self.test_workflow(loader, domain_key)
                    results[domain_key + '_acc'] = acc

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    commons.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                commons.print_row([results[key] for key in results_keys], colwidth=12)

                results.update({
                    'hparams': self.hparams,
                    'args': vars(self.flags)
                })

                epochs_path = os.path.join(self.checkpoint_path, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                self.algorithm_dict = self.algorithm.state_dict()
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

            inputs = []
            labels = []
            domains = []

            for domain_key in self.test_workflow_iters.keys():

                list_task_predictions_per_domain = []
                list_task_targets_per_domain = []
                list_domain_predictions_per_domain = []
                list_domain_targets_per_domain = []

                for inputs, labels, domains in self.test_workflow_iters[domain_key]:

                    n_total += inputs.size(0)

                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    domains = domains.cuda()

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
