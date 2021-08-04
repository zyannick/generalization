import models.dm_algorithms as dm

def get_model(flags):
    model_object = None
    if flags.dm_model == 'arm':
        model_object = dm.ARM.ARM()


    return model_object