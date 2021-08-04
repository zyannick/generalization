import argparse


def g2dm_parser():
    parser = argparse.ArgumentParser(description='RP for domain generalization')
    parser.add_argument('--original_mode', type=str,
                        default='rgb', help='raw or flow')
    parser.add_argument('--middle_mode', type=str,
                        default='flow', help='rgb or flow')
    parser.add_argument('--save_model', default='checkpoint', type=str)
    parser.add_argument('--edge_type', default='no_edge', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=8, type=int)
    parser.add_argument('--nb_frames', default=16, type=int)
    parser.add_argument('--blur_kernel', type=int, default=0)
    parser.add_argument('--operator_kernel', type=int, default=3)
    parser.add_argument('--number_of_domain', type=int, default=3)
    parser.add_argument('--domains_list', type=list,
                        default=['raw_rgb', 'sobel_0_3', 'laplace_0_3'])
    parser.add_argument('--model_name', default='i3d', type=str)

    parser.add_argument('--root', default='../five_fps_cme_sep/', type=str)
    parser.add_argument('--dataset_name', default='cme_sep', type=str)
    parser.add_argument('--split_file', default='./maj_cmd_fall.json', type=str)

    parser.add_argument(
        '--target_root', default='./baga_balanced/videos/', type=str)
    parser.add_argument('--target_split_file',
                        default='./baga_balanced/baga.json', type=str)

    parser.add_argument('--data_aug', type=bool, default=True)
    parser.add_argument('--flow_net', type=bool, default=True)
    parser.add_argument('--continue_training', type=bool, default=True)
    parser.add_argument('--affine_transform', type=bool, default=True)
    parser.add_argument('--is_extended', type=bool, default=False)
    parser.add_argument('--is_inference', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=bool, default=False)
    parser.add_argument('--distance_transform', type=int, default=0)
    parser.add_argument('--type_img', type=str, default='visible')
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    #parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr-task', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr-domain', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr-threshold', type=float, default=1e-4, metavar='LRthrs',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum-task', type=float, default=0.9, metavar='m',
                        help='momentum (default: 0.9)')
    parser.add_argument('--momentum-domain', type=float, default=0.9, metavar='m',
                        help='momentum (default: 0.9)')
    parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                        help='Weight decay coefficient (default: 0.00001')
    parser.add_argument('--factor', type=float, default=0.1, metavar='f',
                        help='LR decrease factor (default: 0.1')
    parser.add_argument('--checkpoint-epoch', type=int, default=8, metavar='N',
                        help='epoch to load for checkpointing. If None, training starts from scratch')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint', metavar='Path',
                        help='Path for checkpointing')
    parser.add_argument('--data-path', type=str, default='../data/pacs/prepared_data/', metavar='Path',
                        help='Data path')

    parser.add_argument('--source1', type=str, default='sobel_0_3',
                        metavar='Path', help='Path to source1 file')
    parser.add_argument('--source2', type=str, default='sobel_3_5',
                        metavar='Path', help='Path to source2 file')
    parser.add_argument('--source3', type=str, default='laplace_0_3',
                        metavar='Path', help='Path to source3 file')

    parser.add_argument('--target', type=str, default='tir_sobel',
                        metavar='Path', help='Path to target data')
    parser.add_argument('--seed', type=int, default=None,
                        metavar='S', help='random seed (default: None)')
    parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir',
                        help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
    parser.add_argument('--alpha', type=float, default=0.8, metavar='alpha',
                        help='balance losses to train encoder. Should be within [0,1]')
    parser.add_argument('--rp-size', type=int, default=3000, metavar='rp',
                        help='Random projection size. Should be smaller than 4096')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='number of epochs to wait before reducing lr (default: 20)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        metavar='l', help='Label smoothing (default: 0.2)')
    parser.add_argument('--warmup-its', type=float, default=500,
                        metavar='w', help='LR warm-up iterations (default: 500)')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--save-every', type=int, default=5, metavar='N',
                        help='how many epochs to wait before logging training status. Default is 5')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='Disables GPU use')
    parser.add_argument('--no-logging', action='store_true',
                        default=False, help='Deactivates logging')
    parser.add_argument('--ablation', choices=['all', 'RP', 'no', 'no_aug', 'all_multidomain', 'all_multidomain_no_aug'], 
                        default='all_multidomain_no_aug',
                        help='Ablation study (removing only RPs (option: RP), RPs+domain classifier (option: all), (default: no))')
    parser.add_argument('--train-mode', choices=['hv', 'avg'], default='hv',
                        help='Train mode (options: hv, avg), (default: hv))')
    parser.add_argument('--train-model', choices=['alexnet', 'resnet18', 'i3d', 'r2p1d'], default='i3d',
                        help='Train model (options: alexnet, resnet18, i3d, r2p1d), (default: i3d))')

    parser.add_argument('--n-runs', type=int, default=1,
                        metavar='n', help='Number of repetitions (default: 3)')


    '''
    epi agg'''
    parser.add_argument("--seed", type=int, default=1, help="")
    parser.add_argument("--test_every", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--num_classes", type=int, default=10, help="")
    parser.add_argument("--step_size", type=int, default=1, help="")
    parser.add_argument("--bn_eval", type=int, default=1, help="")
    parser.add_argument("--loops_train", type=int, default=200000, help="")
    parser.add_argument("--unseen_index", type=int, default=0, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help='')
    parser.add_argument("--weight_decay", type=float, default=0.00005, help='')
    parser.add_argument("--momentum", type=float, default=0.9, help='')
    parser.add_argument("--logs", type=str, default='logs/', help='')
    parser.add_argument("--model_path", type=str, default='', help='')
    parser.add_argument("--state_dict", type=str, default='',  help='')
    parser.add_argument("--data_root", type=str, default='', help='')
    parser.add_argument("--deterministic", type=bool, default=False, help='')

    '''
    epi fcr'''
    parser.add_argument("--test_every", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=20, help="")
    parser.add_argument("--num_classes", type=int, default=10, help="")
    parser.add_argument("--step_size", type=int, default=10000, help="")
    parser.add_argument("--loops_train", type=int, default=200000, help="")
    parser.add_argument("--loops_warm", type=int, default=200000, help="")
    parser.add_argument("--loops_agg_warm", type=int, default=200000, help="")
    parser.add_argument("--unseen_index", type=int, default=0, help="")
    parser.add_argument("--ite_train_epi_c", type=int, default=200000, help="")
    parser.add_argument("--ite_train_epi_f", type=int, default=200000,help="")
    parser.add_argument("--bn_eval", type=int, default=1, help="")
    parser.add_argument("--warm_up_agg", type=int, default=-111, help='')
    parser.add_argument("--lr", type=float, default=0.0001, help='')
    parser.add_argument("--weight_decay", type=float, default=0.00005, help='')
    parser.add_argument("--momentum", type=float, default=0.9, help='')
    parser.add_argument("--loss_weight_epif", type=float, default=0.8, help='')
    parser.add_argument("--loss_weight_epic", type=float, default=0.8, help='')
    parser.add_argument("--loss_weight_epir", type=float, default=0.8, help='')
    parser.add_argument("--logs", type=str, default='logs/', help='')
    parser.add_argument("--model_path", type=str, default='', help='')
    parser.add_argument("--state_dict", type=str, default='', help='')
    parser.add_argument("--data_root", type=str, default='', help='')
    parser.add_argument("--deterministic", type=bool, default=False, help='')


    

    return parser