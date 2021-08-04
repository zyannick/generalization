import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from data_helpers.dataset import Video_Datasets
import utils
import torchvision.models as models_tv
#from data_loader import Loader_source, Loader_validation, Loader_unif_sampling
from train_loop import TrainLoop
import models as models
import data_helpers.videotransforms as videotransforms
import pandas
import PIL
from torchvision import transforms
from torchvision import datasets
import torch.utils.data
import torch.optim as optim
import argparse
import sys
import random
from tqdm import tqdm
import setproctitle
import colorama
from utils import system_info


from args_parser import g2dm_parser

parser = g2dm_parser()

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
args.logging = True if not args.no_logging else False

assert args.alpha >= 0. and args.alpha <= 1.

print('Source domains: {}, {}, {}'.format(
    args.source1, args.source2, args.source3))
print('Orignal mode :{}'.format(args.original_mode))
print('Middle mode :{}'.format(args.middle_mode))
print('Target domain:', args.target)
print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR task: {}'.format(args.lr_task))
print('LR domain: {}'.format(args.lr_domain))
print('L2: {}'.format(args.l2))
print('Alpha: {}'.format(args.alpha))
print('Momentum task: {}'.format(args.momentum_task))
print('Momentum domain: {}'.format(args.momentum_domain))
print('Nadir slack: {}'.format(args.nadir_slack))
print('RP size: {}'.format(args.rp_size))
print('Patience: {}'.format(args.patience))
print('Smoothing: {}'.format(args.smoothing))
print('Warmup its: {}'.format(args.warmup_its))
print('LR factor: {}'.format(args.factor))
print('Ablation: {}'.format(args.ablation))
print('Train mode: {}'.format(args.train_mode))
print('Train model: {}'.format(args.train_model))
print('Seed: {}'.format(args.seed))



argparse_dict = vars(args)

print('\n\n')
for cle, values in argparse_dict.items():
    print(cle +  ' --> ' + str(values))
print('\n\n')





acc_runs = []
acc_blind = []
seeds = [1, 10, 100]

args.n_runs = 1



for run in range(args.n_runs):
    print('Run {}'.format(run))

    train_mode = ''

    print('data augmentation')
    print(args.data_aug)

    # Setting seed
    if args.seed is None:
        random.seed(seeds[run])
        torch.manual_seed(seeds[run])
        if args.cuda:
            torch.cuda.manual_seed(seeds[run])
        checkpoint_path = os.path.join(
            args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(seeds[run]))
    else:
        seeds[run] = args.seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        checkpoint_path = os.path.join(
            args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(args.seed))

    system_info()

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    nb_workers = 0

    print(args.batch_size)

    setproctitle.setproctitle(train_mode)

   


