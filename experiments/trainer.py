import logging
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from Models import utils, analysis

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


class BaseTrainer(object):

    def __init__(self, model, pre_train_loader, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        self.model = model
        self.pre_train_loader = pre_train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = config['device']
        self.optimizer = config['optimizer']
        self.loss_module = config['loss_module']
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.print_conf_mat = print_conf_mat
        self.epoch_metrics = OrderedDict()
        self.save_path = config['output_dir']

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.dataloader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)





class Self_Supervised_Trainer_rASR(BaseTrainer):
    def __init__(self, *args, **kwargs):

        super(Self_Supervised_Trainer_rASR, self).__init__(*args, **kwargs)
        if kwargs['print_conf_mat']:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
        self.mse = nn.MSELoss(reduction='none')
        self.gap = nn.AdaptiveAvgPool1d(1)

    def train_epoch(self, epoch_num=None):
        self.model.copy_weight()
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.pre_train_loader):
            X, targets, X_clean, IDs = batch
            rep_mask, rep_mask_prediction, rec_clean, X_Norm = self.model.pretrain_forward(X.to(self.device),X_clean.to(self.device))

            # rASR_loss = torch.tensor(1e-6, requires_grad=True)
            rASR_loss = F.mse_loss(X_Norm, rec_clean)
            # align_loss = torch.tensor(1e-6, requires_grad=True)
            # align_loss = contrastive_bce_loss(rep_mask, rep_mask_prediction)
            align_loss = F.mse_loss(rep_mask, rep_mask_prediction)

            y = self.gap(rep_mask_prediction.transpose(2, 1)).squeeze()
            y = y - y.mean(dim=0)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_y))
            cov_y = (y.T @ y) / (len(targets) - 1)
            cov_loss = off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])
            total_loss = align_loss + std_loss + cov_loss + rASR_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.model.momentum_update()
            total_samples += 1
            epoch_loss += total_loss.item()
        epoch_loss /= total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['align'] = align_loss
        self.epoch_metrics['std'] = std_loss
        self.epoch_metrics['cov'] = cov_loss
        self.epoch_metrics['rASR_loss'] = rASR_loss

        if (epoch_num + 1) % 5 == 0:
            self.model.eval()
            train_repr, train_labels = make_representation(self.model, self.train_loader)
            test_repr, test_labels = make_representation(self.model, self.test_loader)
            clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            y_hat = clf.predict(test_repr.cpu().detach().numpy())
            acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
            # plot_tSNE(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
            print('Test_acc:', acc_test)
            result_file = open(self.save_path + '/linear_result.txt', 'a+')
            print('{0}, {1}, {2}, {3}, {4}, {5}'.format(int(epoch_num), acc_test, align_loss, std_loss, cov_loss, rASR_loss),
                  file=result_file)
            result_file.close()

        return self.epoch_metrics, self.model

    

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


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