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
from utils import commons, system_info
import hparams_registry
import json
from model_loader import *

from args_parser import global_parser

parser = global_parser()

flags = parser.parse_args()
flags.cuda = True if not flags.no_cuda and torch.cuda.is_available() else False
flags.logging = True if not flags.no_logging else False

assert flags.alpha >= 0. and flags.alpha <= 1.




argparse_dict = vars(flags)


seeds = [1, 10, 100]

flags.n_runs = 1



def running(run):

    print('Run {}'.format(run))

    train_mode = ''

    if flags.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(flags.algorithm, flags.dataset)
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

    system_info()

    transformations = {}

    if flags.input == 'video':
        transformations['train'] = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(),
                                            ])
        transformations['val'] = transforms.Compose([videotransforms.CenterCrop(224)])
        transformations['test'] = transforms.Compose([videotransforms.CenterCrop(224)])
    else:
        transformations['train'] = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(),
                                            ])
        transformations['val'] = transforms.Compose([videotransforms.CenterCrop(224)])
        transformations['test'] = transforms.Compose([videotransforms.CenterCrop(224)])


    train_data = {}
    val_data = {}
    

    for domain_key in flags.source_domains_list:
        train_data[domain_key] = Video_Datasets(data_root=flags.data_root,  split='train', flags=flags, domain_key=domain_key, is_train=True, modality = flags.modality,  transforms=transformations['train'])
        val_data[domain_key] = Video_Datasets(data_root=flags.data_root,  split='val', flags=flags, domain_key=domain_key, is_train=False, modality = flags.modality,  transforms=transformations['val'])

    test_data = {}

    for domain_key in flags.target_domains_list:
        test_data[domain_key] = Video_Datasets(data_root=flags.target_root,  split='val', flags=flags, domain_key=domain_key, is_train=True, modality = 'thermal',  transforms=transformations['val'])

    datasets = {}
    datasets['train'] = train_data
    datasets['val'] = val_data
    datasets['test'] = test_data
    
    model_obj = get_model(flags, hparams, datasets)

    print(flags.batch_size)

    setproctitle.setproctitle(train_mode)


    if flags.phase == 'train':
        model_obj.training()
    elif flags.phase == 'extract_features':
        model_obj.extract_features()
    elif flags.phase == 'vizualize':
        model_obj.visualize_features_maps()


def runs():
    for i in range(flags.n_runs):
        print('Run {}'.format(i))
        running(i)


if __name__ == '__main__':

    # need to add argparse
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpus
    system_info()
    runs()



   


