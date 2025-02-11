import torch
import json
from copy import deepcopy
from torch.utils.data import DataLoader 
from tools.utils import dataset_class
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from Models.model_factory import model_factory
from collections import OrderedDict
from sklearn.metrics import f1_score




def Zero_Shot(Data):
    epoch_metrics = OrderedDict()
    # Path to the JSON file
    json_file_path = 'Checkpoints/TUAB/TUAB_configuration.json'
    # json_file_path = 'Checkpoints/STEW_configuration.json'
    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    config['device'] = 'cuda'
    config['num_labels'] = int(max(Data['All_train_label'])) + 1
    config['Data_shape'] = Data['All_train_data'].shape
    model = model_factory(config)
    # Encoder = load_model(model, 'Checkpoints/STEW_Pretrained.pth')
    Encoder = load_model(model, 'Checkpoints/TUAB/TUABmodel_last.pth')

    Encoder.to(config['device'])

    train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    train_rep, train_labels = make_representation(Encoder, train_loader)
    test_rep, test_labels = make_representation(Encoder, test_loader)

    # Convert tensors to numpy arrays if necessary, here assumed to be in numpy format
    train_rep = train_rep.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_rep = test_rep.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    # Create logistic regression model and Fit model on training data
    clf = fit_lr(train_rep, train_labels)
    clf.fit(train_rep, train_labels)
    # Predict on test data
    y_hat = clf.predict(test_rep)
    y_prob = clf.predict_proba(test_rep)[:, 1]  # Get the probabilities for the positive class
    # Calculate accuracy
    acc_test = accuracy_score(test_labels, y_hat)
    # Calculate AUROC
    if len(set(test_labels)) > 2:
        auroc_test = f1_score(test_labels, y_hat, average='weighted')
    else:
        auroc_test = roc_auc_score(test_labels, y_prob)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels, y_hat)
    print("Confusion Matrix:")
    print(cm)
    epoch_metrics['Accuracy'] = acc_test
    epoch_metrics['AUROC'] = auroc_test

    return epoch_metrics
'''

def Zero_Shot(Data):
    epoch_metrics = OrderedDict()
    # Path to the JSON file
    json_file_path = 'Checkpoints/TUAB/TUAB_configuration.json'
    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    config['device'] = 'cuda'

    config['num_labels'] = int(max(Data['Y'])) + 1
    config['Data_shape'] = Data['X'].shape
    model = model_factory(config)
    Encoder = load_model(model, 'Checkpoints/TUAB/TUABmodel_last.pth')
    Encoder.to(config['device'])
    unique_ids = set(Data['id'])

    accuracies = []
    aurocs = []
    for subject_id in unique_ids:
        print(f"Training with subject {subject_id} left out.")
        # Split the data into training and testing sets based on the subject ID
        train_indices = [i for i, id in enumerate(Data['id']) if id != subject_id]
        test_indices = [i for i, id in enumerate(Data['id']) if id == subject_id]

        train_data = Data['X'][train_indices]
        train_labels = Data['Y'][train_indices]
        test_data = Data['X'][test_indices]
        test_labels = Data['Y'][test_indices]

        train_dataset = dataset_class(train_data, train_labels, config)
        test_dataset = dataset_class(test_data, test_labels, config)

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

        train_rep, train_labels = make_representation(Encoder, train_loader)
        test_rep, test_labels = make_representation(Encoder, test_loader)

        # Convert tensors to numpy arrays if necessary, here assumed to be in numpy format
        train_rep = train_rep.cpu().detach().numpy()
        train_labels = train_labels.cpu().detach().numpy()
        test_rep = test_rep.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()

        # Create logistic regression model and Fit model on training data
        clf = fit_lr(train_rep, train_labels)
        clf.fit(train_rep, train_labels)

        # Predict on test data
        y_hat = clf.predict(test_rep)
        y_prob = clf.predict_proba(test_rep)[:, 1]  # Get the probabilities for the positive class

        # Calculate accuracy
        acc_test = accuracy_score(test_labels, y_hat)

        # Calculate AUROC
        if len(set(test_labels)) > 2:
            auroc_test = f1_score(test_labels, y_hat, average='weighted')
        else:
            auroc_test = roc_auc_score(test_labels, y_prob)
        
        print(f'Test_acc for subject {subject_id}:', acc_test)
        cm = confusion_matrix(test_labels, y_hat)
        print("Confusion Matrix:")
        print(cm)

        epoch_metrics[f'Accuracy_subject_{subject_id}'] = acc_test
        epoch_metrics[f'AUROC_subject_{subject_id}'] = auroc_test
        accuracies.append(acc_test)
        aurocs.append(auroc_test)
    # Calculate the average accuracy over all subjects
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_aurocs = sum(aurocs) / len(aurocs)

    print(f'Average accuracy over all subjects: {avg_accuracy}')
    print(f'Average AUROC over all subjects: {avg_aurocs}')
    epoch_metrics['Average_Accuracy'] = avg_accuracy
    epoch_metrics['Average_AUROC'] = avg_aurocs

    return epoch_metrics
'''



def load_model(model, model_path, optimizer=None, resume=False, change_output=True,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    keys_to_remove = ['Layer_Norm', 'predict_head', 'PositionalEncoding']
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if any(key.startswith(k) for k in keys_to_remove):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def make_representation(model, data):
    out = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data):
            X, targets, IDs = batch
            rep = model.linear_prob(X.to('cuda'))
            # out_rep = torch.mean(rep, dim=1)
            out.append(rep)
            labels.append(targets)

        out = torch.cat(out, dim=0)
        labels = torch.cat(labels, dim=0)
    return out, labels


def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe