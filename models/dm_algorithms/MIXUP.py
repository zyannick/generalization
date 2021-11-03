import os

from utils.commons import random_pairs_of_minibatches

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.nn.functional as F

import utils.commons as commons
import time
from models.dm_algorithms import ERM

import numpy as np




class Mixup(ERM):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance):
        super(Mixup, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, class_balance)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in commons.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}