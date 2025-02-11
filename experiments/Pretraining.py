
from experiments.EEG2Rep_Pretraining import EEG2Rep_Pretraining
from experiments.EEGX_Pretraining import EEGX_Pretraining
from experiments.Biot_Pretraining import Biot_Pretraining
from experiments.MAEEG_Pretraining import MAEEG_Pretraining


def Pretraining_Finetuning(config, Data):
    if config['Model'] == 'EEG2Rep':
        best_aggr_metrics_test, all_metrics = EEG2Rep_Pretraining(config, Data)
    elif config['Model'] == 'EEG-X':   
        best_aggr_metrics_test, all_metrics = EEGX_Pretraining(config, Data)
    elif config['Model'] == 'Biot':
        best_aggr_metrics_test, all_metrics = Biot_Pretraining(config, Data)
    elif config['Model'] == 'MAEEG':
        best_aggr_metrics_test, all_metrics = MAEEG_Pretraining(config, Data)
    else:
        raise ValueError("Model not implemented")   

    return best_aggr_metrics_test, all_metrics 