from models.dm_algorithms.AbstractDANN import AbstractDANN


class CDANN(AbstractDANN):
    def __init__(self, flags, hparams, input_shape, datasets, checkpoint_path, class_balance, conditional):
        super(CDANN, self).__init__(flags, hparams, input_shape, datasets, checkpoint_path, conditional=True, class_balance=True)
