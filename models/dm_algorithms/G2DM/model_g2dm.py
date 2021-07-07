

from data_helpers.dataset import Video_Datasets, Target_Datasets
from .g2dm import G2DM
from ..backbones import models as models
import torch.utils.data
import torch.optim as optim





def launch_model(launch_mode, source_dataset, validation_dataset, target_dataset, args):

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

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



    acc_runs = []
    acc_blind = []



    

    trainer = G2DM(models_dict, optimizer_task, source_loader, test_source_loader, target_loader, args.nadir_slack,
                        args.alpha, args.patience, args.factor, args.smoothing, args.warmup_its, args.lr_threshold, args, batch_size=args.batch_size,
                        checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda,
                        ablation=args.ablation, logging=args.logging, train_mode=args.train_mode)
    err, err_blind = trainer.train(n_epochs=args.epochs, save_every=1)

    acc_runs.append(1 - err)
    acc_blind.append(err_blind)

