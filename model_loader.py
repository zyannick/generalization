from models.algoritms import DefaultModel
from models.dm_algorithms import *

model_dict = {
    'arm': ARM,
    'cdann': CDANN, 
    'coral': CORAL,
    'erm': ERM,
    'epi_fcr': ModelEpiFCR,
    'epi_fcr_agg': ModelAggregate,
    'fish': Fish,
    'g2dm': G2DM,
    'irm': IRM,
    'mixup': Mixup,
    'vrex': VREX
}

def get_model(flags, hparams, datasets) -> DefaultModel:

    model_object = None
    if flags.dm_model in model_dict.keys():
        model_object = model_dict[flags.dm_model](flags, hparams, flags.input_shape, datasets, flags.checkpoint_path, flags.class_balance)
    else:
        raise ValueError('No model found for {}'.format(flags.dm_model))

    return model_object