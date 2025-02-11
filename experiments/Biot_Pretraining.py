import os 
import logging
import torch
from torch.utils.data import DataLoader 
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import OrderedDict
from Models import utils, analysis
from experiments.Supervise import SupervisedTrainer
from experiments.train_pipeline import selfsupervised_pipeline, supervise_pipeline
from tools.utils import dataset_class
from Models.model_factory import model_factory, count_parameters
from Models.utils import load_model
from Models.loss import get_loss_module
from Models.optimizers import get_optimizer

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less


def Biot_Pretraining(config, Data):
    # ---------------------------------------- Self Supervised Data loader -------------------------------------------
    pre_train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config)
    pre_train_loader = DataLoader(dataset=pre_train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    # -------------------------------------------- Build Model ---------------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    config['Data_shape'] = Data['All_train_data'].shape
    config['num_labels'] = int(max(Data['All_train_label'])) + 1
    Encoder = model_factory(config)
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = Biot_SelfSupervised_Trainer(Encoder, pre_train_loader, train_loader, test_loader, config, l2_reg=0, print_conf_mat=False)
    selfsupervised_pipeline(config, Encoder, SS_trainer, save_path)


    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    # ---------------------------------------- Linear Probing -------------------------------------------------------------
    train_repr, train_labels = make_representation(SS_Encoder, train_loader)
    test_repr, test_labels = make_representation(SS_Encoder, test_loader)
    clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
    y_hat = clf.predict(test_repr.cpu().detach().numpy())
    # plot_tSNE(test_repr.cpu().detach().numpy(), test_labels.cpu().detach().numpy())
    acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
    print('Test_acc:', acc_test)
    cm = confusion_matrix(test_labels.cpu().detach().numpy(), y_hat)
    print("Confusion Matrix:")
    print(cm)

    # ---------------------------------------- Fine Tuning -------------------------------------------------------------
    
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config)
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config)
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    logger.info('Starting Fine_Tuning...')
    S_trainer = SupervisedTrainer(SS_Encoder, train_loader, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(SS_Encoder, val_loader, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    supervise_pipeline(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, test_loader, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)


    return best_aggr_metrics_test, all_metrics



class Biot_SelfSupervised_Trainer(object):
    def __init__( self, model, pre_train_loader, train_loader, test_loader, config, optimizer=None, l2_reg=None, print_interval=10,
                 console=True, print_conf_mat=False):
        super(Biot_SelfSupervised_Trainer, self).__init__()
        self.analyzer = analysis.Analyzer(print_conf_mat=False)
        if print_conf_mat:
            self.analyzer = analysis.Analyzer(print_conf_mat=True)
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
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.T = 0.2


    def print_callback(self, i_batch, metrics, prefix=''):
        total_batches = len(self.train_loader)
        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


    def train_epoch(self, epoch_num=None):
        # self.model.copy_weight()
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        for i, batch in enumerate(self.pre_train_loader):
            X, targets, IDs = batch

            prest_masked_emb, prest_samples_emb = self.model.pretrain_forward(X.to(self.device))
            # L2 normalize
            prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
            prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
            N = X.shape[0]

            # representation similarity matrix, NxN
            logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
            labels = torch.arange(N).to(logits.device)
            total_loss = F.cross_entropy(logits, labels, reduction="mean")

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_samples += 1
            epoch_loss += total_loss.item()
        epoch_loss /= total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss


        if (epoch_num + 1) % 5 == 0:
            self.model.eval()
            train_repr, train_labels = make_representation(self.model, self.train_loader)
            test_repr, test_labels = make_representation(self.model, self.test_loader)
            clf = fit_lr(train_repr.cpu().detach().numpy(), train_labels.cpu().detach().numpy())
            y_hat = clf.predict(test_repr.cpu().detach().numpy())
            acc_test = accuracy_score(test_labels.cpu().detach().numpy(), y_hat)
            print('Test_acc:', acc_test)
            result_file = open(self.save_path + '/linear_result.txt', 'a+')
            print(f'{int(epoch_num)}, {acc_test}', file=result_file)
            result_file.close()

        return self.epoch_metrics, self.model


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