import os
import numpy as np
import logging
from sklearn import model_selection

logger = logging.getLogger(__name__)

def load(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '_Filtered.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '_Filtered.npy', allow_pickle=True)
        if np.any(Data_npy['val_data']):
            Data['train_data'] = Data_npy['train_data']
            Data['train_label'] = Data_npy['train_label']
            Data['val_data'] = Data_npy['val_data']
            Data['val_label'] = Data_npy['val_label']
            Data['test_data'] = Data_npy['test_data']
            Data['test_label'] = Data_npy['test_label']
            Data['max_len'] = Data['train_data'].shape[2]
            Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
            Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
            if config['Evaluation'] == 'Cross-domain':
                Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data_npy)
                logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy['train_data'], Data_npy['train_label'], 0.1)
            Data['All_train_data'] = Data_npy['train_data']
            Data['All_train_label'] = Data_npy['train_label']
            Data['test_data'] = Data_npy['test_data']
            Data['test_label'] = Data_npy['test_label']
            Data['max_len'] = Data['train_data'].shape[2]
    if config['Model'] == 'EEG-X':
        clean_npy = np.load(config['data_dir'] + '/' + config['problem'] + '_Clean.npy', allow_pickle=True)
        Data['train_data_clean'] = np.concatenate((clean_npy['train_data'], clean_npy['val_data']))

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def load_TUEV(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/TUEV_train.npy'):
        logger.info("Loading preprocessed data ...")
        Train_data = np.load(config['data_dir'] + '/TUEV_train.npy', allow_pickle=True)
        Val_data = np.load(config['data_dir'] + '/TUEV_val.npy', allow_pickle=True)
        Test_data = np.load(config['data_dir'] + '/TUEV_test.npy', allow_pickle=True)

        Data['train_data'] = Train_data['X']
        Data['train_label'] = Train_data['y']
        Data['val_data'] = Val_data['X']
        Data['val_label'] = Val_data['y']
        Data['test_data'] = Test_data['X']
        Data['test_label'] = Test_data['y']
        Data['max_len'] = Data['train_data'].shape[2]
        Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
        Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
        if config['Evaluation'] == 'Cross-domain':
            Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data)
            logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
    if config['Model'] == 'EEG-X':
        Data['train_data_clean'] = np.concatenate((Train_data['X_clean'], Val_data['X_clean']))
    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def load_BCICIV_2a(config):
    # Build data
    '''
    Data = np.load(config['data_dir'] +'/ICA_BCICIV_2a.npy', allow_pickle=True)
    '''
    Data = np.load(config['data_dir'] +'/ICA_BCICIV_2a_splited.npy', allow_pickle=True)
    Data['All_train_data'] = Data['train_data']
    Data['All_train_label'] = Data['train_label']

    Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data['train_data'], Data['train_label'], 0.1)
    

    return Data


def load_crowdsourced(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True).item()
        if np.any(Data_npy['val_data']):
            Data['train_data'] = Data_npy['train_data']
            Data['train_label'] = Data_npy['train_label']
            Data['val_data'] = Data_npy['val_data']
            Data['val_label'] = Data_npy['val_label']
            Data['test_data'] = Data_npy['test_data']
            Data['test_label'] = Data_npy['test_label']
            Data['max_len'] = Data['train_data'].shape[2]
            Data['All_train_data'] = np.concatenate((Data['train_data'], Data['val_data']))
            Data['All_train_label'] = np.concatenate((Data['train_label'], Data['val_label']))
            

    logger.info("{} samples will be used for zero-shot training".format(len(Data['All_train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def load_npy(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True)

        if np.any(Data_npy.item().get('val_data')):
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['val_data'] = Data_npy.item().get('val_data')
            Data['val_label'] = Data_npy.item().get('val_label')
            Data['All_train_data'] = Data_npy.item().get('All_train_data')
            Data['All_train_label'] = Data_npy.item().get('All_train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[1]
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy.item().get('train_data'), Data_npy.item().get('train_label'), 0.1)
            Data['All_train_data'] = Data_npy.item().get('train_data')
            Data['All_train_label'] = Data_npy.item().get('train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[2]

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def Cross_Domain_loader(domain_data):
    All_train_data = domain_data.item().get('All_train_data')
    All_train_label = domain_data.item().get('All_train_label')
    # Load DREAMER for Pre-Training
    DREAMER = np.load('Dataset/DREAMER/DREAMER.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, DREAMER.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, DREAMER.item().get('All_train_label')), axis=0)

    # Load Crowdsource for Pre-Training
    Crowdsource = np.load('Dataset/Crowdsource/Crowdsource.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, Crowdsource.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, Crowdsource.item().get('All_train_label')), axis=0)
    return All_train_data, All_train_label


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label