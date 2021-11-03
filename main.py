from models.backbones.r2p1d import load_model
import os

from data_helpers.dataset import Video_Datasets
import data_helpers.videotransforms as videotransforms
from torchvision import transforms
from torchvision import datasets
import torch.utils.data
import torch.optim as optim
import random
from tqdm import tqdm
import numpy as np
import setproctitle
from utils import commons

import hparams_registry
import json
from model_loader import *


from args_parser import global_parser

parser = global_parser()

flags = parser.parse_args()
flags.cuda = True if flags.cuda and torch.cuda.is_available() else False
flags.logging = True

# assert flags.alpha >= 0. and flags.alpha <= 1.




argparse_dict = vars(flags)


seeds = [1, 10, 100]

flags.n_runs = 1



def running(run):

    print('Run {}'.format(run))

    train_mode = ''

    if flags.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(flags.algorithm)
    else:
        hparams = hparams_registry.random_hparams(flags.algorithm, flags.dataset,
            commons.seed_hash(flags.hparams_seed, flags.trial_seed))
    if flags.hparams:
        hparams.update(json.loads(flags.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(flags.seed)
    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        flags.device = "cuda"
    else:
        flags.device = "cpu"

    # Setting seed
    if flags.seed is None:
        random.seed(seeds[run])
        torch.manual_seed(seeds[run])
        if flags.cuda:
            torch.cuda.manual_seed(seeds[run])
        checkpoint_path = os.path.join(
            flags.checkpoint_path, flags.target + '_' + train_mode + '_seed' + str(seeds[run]))
    else:
        seeds[run] = flags.seed
        random.seed(flags.seed)
        torch.manual_seed(flags.seed)
        if flags.cuda:
            torch.cuda.manual_seed(flags.seed)
        checkpoint_path = os.path.join(
            flags.checkpoint_path, flags.target + '_' + train_mode + '_seed' + str(flags.seed))

    commons.system_info()

    transformations = {}

    if flags.input == 'video':
        transformations['train'] = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(),
                                            ])
        transformations['val'] = transforms.Compose([videotransforms.CenterCrop(224)])
        transformations['test'] = transforms.Compose([videotransforms.CenterCrop(224)])
    else:
        transformations['train'] = None
        transformations['val'] = None
        transformations['test'] = None


    train_data = {}
    val_data = {}
    

    for domain_key in flags.source_domains_list:
        train_data[domain_key] = Video_Datasets(data_root=flags.data_root, source_target= flags.source,  split='training', flags=flags, domain_key=domain_key, is_train=True, modality = flags.modality,  transforms=transformations['train'])
        val_data[domain_key] = Video_Datasets(data_root=flags.data_root, source_target= flags.source,  split='testing', flags=flags, domain_key=domain_key, is_train=False, modality = flags.modality,  transforms=transformations['val'])

    test_data = {}

    for domain_key in flags.target_domains_list:
        test_data[domain_key] = Video_Datasets(data_root=flags.data_root, source_target= flags.target, split='testing', flags=flags, domain_key=domain_key, is_train=True, modality = 'thermal',  transforms=transformations['val'])

    datasets = {}
    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data
    class_idx = {}

    flags.input_shape = 224
    flags.class_balance = True

    
    algorithm_class = get_algorithm(flags)
    launcher_class = get_launcher(flags)

    launcher_object = launcher_class(flags.dm_model, algorithm_class, flags, hparams, flags.input_shape, datasets, class_idx, flags.checkpoint_path, flags.class_balance)

    launcher_object.train_workflow()

    setproctitle.setproctitle(train_mode)



def runs():
    for i in range(flags.n_runs):
        running(i)


if __name__ == '__main__':
    # need to add argparse
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpus
    commons.system_info()
    runs()



   


