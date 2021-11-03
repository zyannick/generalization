import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def global_parser():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_mode', type=str,
                        default='rgb', help='raw, flow, idt')
    parser.add_argument('--middle_transform', type=str,
                        default=None, help='flow, None')
    parser.add_argument('--checkpoint_path', default='checkpoint', type=str)
    parser.add_argument('--dataset', default='CME', type=str)
    parser.add_argument('--data_root', default='Datasets', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=8, type=int)
    parser.add_argument('--nb_frames', default=16, type=int)
    parser.add_argument('--number_of_domain', type=int, default=3)
    parser.add_argument('--source_domains_list', type=list,
                        default=['raw_rgb', 'sobel_0_3', 'laplace_0_3'])
    parser.add_argument('--target_domains_list',
                        type=list,  default=['raw_rgb'])
    parser.add_argument('--feature_backbone', default='i3d',
                        type=str, help='i3d, r2p1d, x3d, vit')
    parser.add_argument('--source', default='cme', type=str)
    parser.add_argument('--target', default='baga', type=str)

    parser.add_argument('--dm_model', default='g2dm', type=str, choices=[None,  'arm', 'cdann', 'coral', 'erm', 'epi_fcr', 'epi_fcr_agg',
                                                                       'fish', 'g2dm', 'irm', 'mixup'])

    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')

    parser.add_argument('--launch_mode', type=str, default="train",
                        help="train, infer", choices=['train', 'infer'])
    parser.add_argument('--modality', type=str, default="visible",
                        help="The input of the model", choices=['visible', 'edge', 'tir'])
    parser.add_argument('--continue_training',
                        type=boolean_string, default=True)

    # data setting
    parser.add_argument('--affine_transform',
                        type=boolean_string, default=True)
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--video_augmentations',
                        type=boolean_string, default=True)

    parser.add_argument('--input', type=str, default='video',
                        choices=['video', 'idt'])

    # train settings
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--checkpoint-epoch', type=int, default=8, metavar='N',
                        help='epoch to load for checkpointing. If None, training starts from scratch')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint',
                        metavar='Path', help='Path for checkpointing')
    parser.add_argument('--label_smoothing',
                        type=boolean_string, default=False)

    return parser
