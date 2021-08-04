from models.dm_algorithms.AbstractDANN import AbstractDANN


class DANN(AbstractDANN):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance, conditional):
        super(DANN, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, conditional=False, class_balance=False)
