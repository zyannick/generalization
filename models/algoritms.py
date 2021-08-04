import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import utils.commons as commons
import multiprocessing as mp
import os
import torch.cuda as cuda

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError


    def predict(self, x):
        raise NotImplementedError


class DefaultModel(torch.nn.Module):

    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(DefaultModel, self).__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.class_balance = class_balance
        self.input_shape = input_shape
        # self.backbone = gears['backbone']
        # self.discriminator = gears['disc']
        # self.classifier = gears['classifier']
        self.flags = flags
        self.num_classes = flags.num_classes
        self.num_domains  = flags.num_domains
        self.hparams = hparams
        self.datasets = datasets
        self.checkpoint_path = checkpoint_path
        self.num_devices = cuda.device_count()
        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path,
                                                'checkpoint_{}ep.pt')
        self.update_count = 0
        self.setup()
        self.setup_path()
        self.configure()

    def setup(self):
        raise NotImplementedError

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

    def load_checkpoint(self, epoch):
        raise NotImplementedError

    def configure(self):
        raise NotImplementedError

    def training(self):
        raise NotImplementedError

    def testting(self):
        raise NotImplementedError

    def extract_features(self):
        raise NotImplementedError

    def vizualize_features(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError



