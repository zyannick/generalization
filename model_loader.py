from models.dm_algorithms import *
import models.algoritms as algo
from models.launchers import *


algorithm_dict = {
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

launcher_dict = {
    None: DefaultLauncher,
    'arm': DefaultLauncher,
    'cdann': DefaultLauncher,
    'coral': DefaultLauncher,
    'erm': DefaultLauncher,
    'epi_fcr': EpiFCRLauncher,
    'epi_fcr_agg': DefaultLauncher,
    'fish': DefaultLauncher,
    'g2dm': G2DMLauncher,
    'irm': DefaultLauncher,
    'mixup': DefaultLauncher,
    'vrex': DefaultLauncher
}


def get_algorithm(flags) -> algo.Algorithm:
    if flags.dm_model in algorithm_dict.keys():
        model_class = algorithm_dict[flags.dm_model]
    else:
        raise ValueError('No model found for {}'.format(flags.dm_model))
    return model_class


def get_launcher(flags) -> DefaultLauncher:
    if flags.dm_model in launcher_dict.keys():
        launcher_class = launcher_dict[flags.dm_model]
    else:
        raise ValueError('No launcher found for {}'.format(flags.dm_model))
    return launcher_class
