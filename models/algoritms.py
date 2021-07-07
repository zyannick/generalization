import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

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


class DefaultModel(object):

    def __init__(self, flags,  num_domains, input_shape, backbone = None,  class_balance = False):
        super(DefaultModel, self).__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.class_balance = class_balance
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def setup(self):
        raise NotImplementedError

    def setup_path(self):
        raise NotImplementedError

    def load_state_dict(self):
        raise NotImplementedError

    def configure(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def extract_features(self):
        raise NotImplementedError

    def vizualize_features(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError



