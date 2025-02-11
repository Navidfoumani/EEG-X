import os
import os
import numpy as np
import pandas as pd
import argparse
import logging
# Import Project Modules -----------------------------------------------------------------------------------------------
from tools.utils import Setup, Initialization, Data_Loader, print_title
from experiments.Supervise import Supervise
from experiments.Zeroshot import Zero_Shot
from experiments.Pretraining import Pretraining_Finetuning

# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='1', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
# --------------------------------------------------- I/O --------------------------------------------------------------
parser.add_argument('--data_dir', default='datasets/ICA_BCICIV_2a', help='Data directory')
parser.add_argument('--output_dir', default='Results', help='Root output directory. Time-stamped directories will be created inside.')
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Parameters and Hyperparameters ----------------------------------------------
parser.add_argument('--pretrain_epochs', type=int, default=200, help='Number of pre-training epochs')
parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of fine-tuning or supervise epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout regularization ratio')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--val_interval', type=int, default=5, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy'}, default='loss', help='Metric used for best epoch')
parser.add_argument('--ICA', default=True, type=bool, choices={'True', 'False'}, help='Having ICA reconstruction in loss function or Not')
# -------------------------------------------------- EEG-X ----------------------------------------------------------
parser.add_argument('--Model', default='EEG-X', choices={'EEG-X','EEG2Rep', 'Biot', 'EEGPT', 'LaBraM', 'MAEEG'}, help="Model Type")
parser.add_argument('--Training_mode', default='Zero-Shot', choices={'Supervise_training', 'Pretraining_Finetuning', 'Zero-Shot'}, 
                    help="Training Mode")
parser.add_argument('--Evaluation', default='In-domain', choices={'In-domain', 'Cross-domain'}, help="Evaluation Mode")
parser.add_argument('--Input_Embedding', default='Channel-wise', choices={'Channel-wise', 'CNN'}, help="Input Embedding Architecture")
parser.add_argument('--T_Pos_Encoding', default=['Sin'], choices={'Sin', 'Vector_Embed'}, help="Temporal Position Encoding Method")
parser.add_argument('--C_Pos_Encoding', default=['Location'], choices={'Location', 'Vector_Embed'}, help="Channel info Encoding Method")

parser.add_argument('--layers', type=int, default=4, help="Number of layers for the context/target encoders")
parser.add_argument('--pre_layers', type=int, default=2, help="Number of layers for the Predictor")
parser.add_argument('--mask_ratio', type=float, default=0.5, help=" masking ratio")
parser.add_argument('--momentum', type=float, default=0.99, help="Beta coefficient for EMA update")

parser.add_argument('--patch_size', type=int, default=128, help='Patch size for data segmentation. Data is preprocessed to 128Hz')
parser.add_argument('--patch_stride', type=int, default=32, help='Patch stride for data segmentation')
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of feedforward network of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    config['device'] = Initialization(config)
    print_title(config['problem'])
    logger.info("Loading Data ...")
    Data = Data_Loader(config)
    if config['Training_mode'] == 'Pretraining_Finetuning':
        best_aggr_metrics_test, all_metrics = Pretraining_Finetuning(config, Data)
    elif config['Training_mode'] == 'Supervise_training':
        best_aggr_metrics_test, all_metrics = Supervise(config, Data)
    elif config['Training_mode'] == 'Zero-Shot':
        best_aggr_metrics_test = Zero_Shot(Data)

    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print_title(config['problem'])
    print(print_str)