from Models.EEG_X import EEG_X
from Models.EEG2Rep import EEG2Rep
from Models.Biot import Biot_Pretrain as Biot
from Models.MAEEG import MAEEG


def model_factory(config):
    if config['Model'] == 'EEG-X':
        model = EEG_X(config)
    elif config['Model'] == 'EEG2Rep':
        model = EEG2Rep(config)
    elif config['Model'] == 'Biot':
        model = Biot(config)
    elif config['Model'] == 'LaBraM':
        model = Biot(config)
    elif config['Model'] == 'MAEEG':
        model = MAEEG(config)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)