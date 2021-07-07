from ..base_options import BaseOptions


class AGG_Options(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
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
        return parser
