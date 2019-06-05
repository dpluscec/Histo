"""Model contains functions for training PyTorch model."""

import time
import logging
import copy
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import histo.metrics as metrics
from histo.dataset import TRAIN, VALID, TEST

_LOGGER = logging.getLogger(__name__)


def train(model, loaders_dict, num_epochs, optimizer, criterion, device, hook=None):
    hook_flag = hook is not None

    if hook_flag:
        hook.training_start(model=model, loaders_dict=loaders_dict,
                            criterion=criterion)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in range(0, num_epochs):
        if hook_flag:
            hook.epoch_start(epoch=epoch, num_epochs=num_epochs,
                             loaders_dict=loaders_dict)
        train_loss = run_epoch(model=model, data=loaders_dict[TRAIN], optimizer=optimizer,
                               criterion=criterion, phase=TRAIN, device=device, hook=hook)
        valid_loss = run_epoch(model=model, data=loaders_dict[VALID], optimizer=optimizer,
                               criterion=criterion, phase=VALID, device=device, hook=hook)

        val_conf_mat = evaluate(model=model, data=loaders_dict[VALID], device=device)
        curr_f1 = metrics.f1(confusion_matrix=val_conf_mat)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_model_wts = copy.deepcopy(model.state_dict())

        if hook_flag:
            hook.epoch_end(epoch=epoch, num_epochs=num_epochs, loaders_dict=loaders_dict,
                           model=model, train_loss=train_loss, valid_loss=valid_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)

    if hook_flag:
        hook.training_end(best_model=model)

    return model


def run_epoch(model, data, optimizer, criterion, phase, device, hook=None):
    '''Method trains or evaluates given model for one epoch (one pass through whole data).

    Parameters
    ----------
    model : nn.Module
        model that needs to be trained or evaluated
    data : torch.utils.data.DataLoader
        dataloader used for iterating over data
    optimizer : torch.optim.Optimizer
        model optimizer, None if validation phase
    criterion : loss
        pytorch loss function
    phase : str
        TRAIN or VALID phase id
    device : torch.device
        device on which to perform operations
    hook : TrainingHook
        used for adding functionalities to epoch run
    '''
    hook_flag = hook is not None
    if phase == TRAIN:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    batch_n = 0
    for batch_num, (batch_x, batch_y) in enumerate(data):
        if hook_flag:
            hook.batch_start(phase=phase, batch_num=batch_num, data=data, model=model)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        model.zero_grad()
        with torch.set_grad_enabled(phase == TRAIN):
            loss = None
            if model.__class__.__name__ == "Inception3" and phase == TRAIN:
                outputs, aux_outputs = model(batch_x)
                loss_main = criterion(input=outputs, target=batch_y)
                loss_aux = criterion(input=aux_outputs, target=batch_y)
                loss = loss_main + 0.4*loss_aux
            else:
                logits = model(batch_x)
                loss = criterion(input=logits, target=batch_y)
            running_loss += loss
            if phase == TRAIN:
                loss.backward()
                optimizer.step()
        batch_n = batch_num
        if hook_flag:
            hook.batch_end(phase=phase, batch_num=batch_num, data=data,
                           model=model, batch_loss=loss)
    return running_loss/(batch_n+1)


def evaluate(model, data, device):
    result_mat = np.zeros((2, 2))
    model.eval()
    prob_output = nn.Sigmoid()

    with torch.no_grad():
        for batch_x, batch_y in data:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = prob_output(logits).cpu().numpy()

            y_pred = np.where(probs >= 0.5, 1, 0)
            y_true = batch_y.cpu().numpy()
            curr_cmat = confusion_matrix(y_true, y_pred)
            result_mat += curr_cmat
    return result_mat


def predict_data(model, data, device, return_labels=True):
    """Function calculates model predictions for given data.

    Parameters
    ----------
    model : nn.Module
        PyTorch model instance
    data : torch.utils.data.DataLoader
        dataloader used for iterating over data
    device : torch.device
        device on which to perform operations
    return_labels : bool
        if true function will return target labels from data, otherwise returns only
        predictions

    Returns
    -------
    predictions or (predictions, labels)
        return type is defined with return_labels flag
    """ 
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        prob_output = nn.Sigmoid()
        for batch_x, batch_y in data:
            batch_x = batch_x.to(device)
            if return_labels:
                labels.extend(list(batch_y.numpy()))
            logits = model(batch_x)
            probs = prob_output(logits)
            predictions.extend(list(probs.cpu().numpy()))

    predictions = np.array(predictions)

    if return_labels:
        labels = np.array(labels)
        return (predictions, labels)
    return predictions


class TrainingHook:
    def __init__(self):
        pass

    def training_start(self, model, loaders_dict, criterion):
        pass

    def training_end(self, best_model):
        pass

    def epoch_start(self, epoch, num_epochs, loaders_dict):
        pass

    def epoch_end(self, epoch, num_epochs, loaders_dict, model, train_loss, valid_loss):
        pass

    def batch_start(self, phase, batch_num, data, model):
        if phase == TRAIN:
            self.batch_train_start(batch_num=batch_num, data=data, model=model)
        elif phase == VALID:
            self.batch_valid_start(batch_num=batch_num, data=data, model=model)
        else:
            raise ValueError("Invalid phase.")

    def batch_train_start(self, batch_num, data, model):
        pass

    def batch_valid_start(self, batch_num, data, model):
        pass

    def batch_end(self, phase, batch_num, data, model, batch_loss):
        if phase == TRAIN:
            self.batch_train_end(batch_num=batch_num, data=data,
                                 model=model, batch_loss=batch_loss)
        elif phase == VALID:
            self.batch_valid_end(batch_num=batch_num, data=data,
                                 model=model, batch_loss=batch_loss)
        else:
            raise ValueError("Invalid phase.")

    def batch_train_end(self, batch_num, data, model, batch_loss):
        pass

    def batch_valid_end(self, batch_num, data, model, batch_loss):
        pass


class BasicTrainingHook(TrainingHook):
    def __init__(self):
        super(BasicTrainingHook, self).__init__()
        self.train_start_time = None

    def training_start(self, model, loaders_dict, criterion):
        self.train_start_time = time.time()

    def training_end(self, best_model):
        time_elapsed = time.time()-self.train_start_time
        _LOGGER.info('Training time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def epoch_start(self, epoch, num_epochs, loaders_dict):
        _LOGGER.info('Epoch %s/%s', str(epoch+1), str(num_epochs))
        _LOGGER.info('-' * 10)

    def epoch_end(self, epoch, num_epochs, loaders_dict, model, train_loss, valid_loss):
        _LOGGER.info("Epoch loss - train: %s", train_loss)
        _LOGGER.info("Epoch loss - valid: %s", valid_loss)

    def batch_train_end(self, batch_num, data, model, batch_loss):
        _LOGGER.info("[Batch]: {}/{}, loss {:.5f}".format(
            batch_num + 1, len(data), batch_loss))

    def batch_valid_end(self, batch_num, data, model, batch_loss):
        _LOGGER.info("[Batch]: {}/{}, loss {:.5f}".format(
            batch_num + 1, len(data), batch_loss))


class DetailedMeasurementTrainingHook(BasicTrainingHook):
    def __init__(self, device):
        super(DetailedMeasurementTrainingHook, self).__init__()
        self.device = device
        self.loaders_dict = None
        self.metrics_dict = {
            'train_acc': [],
            'train_f1': [],
            'train_loss': [],
            'valid_acc': [],
            'valid_f1': [],
            'valid_loss': []
            }

    def training_end(self, best_model):
        super(DetailedMeasurementTrainingHook, self).training_end(best_model=best_model)
        for metric_key in self.metrics_dict:
            values_str = "\t".join([str(i) for i in self.metrics_dict[metric_key]])
            _LOGGER.info("%s", metric_key)
            _LOGGER.info("%s", values_str)
        _LOGGER.info("Best model metrics: train, valid, test: acc, f1")
        train_conf_mat = evaluate(model=best_model, data=self.loaders_dict[TRAIN],
                                  device=self.device)
        train_acc = metrics.accuracy(confusion_matrix=train_conf_mat)
        train_f1 = metrics.f1(confusion_matrix=train_conf_mat)
        _LOGGER.info("%s, %s", str(train_acc), str(train_f1))
        val_conf_mat = evaluate(model=best_model, data=self.loaders_dict[VALID],
                                device=self.device)
        val_acc = metrics.accuracy(confusion_matrix=val_conf_mat)
        val_f1 = metrics.f1(confusion_matrix=val_conf_mat)
        _LOGGER.info("%s, %s", str(val_acc), str(val_f1))
        test_conf_mat = evaluate(model=best_model, data=self.loaders_dict[TEST],
                                 device=self.device)
        test_acc = metrics.accuracy(confusion_matrix=test_conf_mat)
        test_f1 = metrics.f1(confusion_matrix=test_conf_mat)
        _LOGGER.info("%s, %s", str(test_acc), str(test_f1))

    def training_start(self, model, loaders_dict, criterion):
        super(DetailedMeasurementTrainingHook, self).training_start(
            model=model, loaders_dict=loaders_dict, criterion=criterion)
        self.loaders_dict = loaders_dict
        _LOGGER.info("start metrics")
        val_conf_mat = evaluate(model=model, data=loaders_dict[VALID], device=self.device)
        val_acc = metrics.accuracy(confusion_matrix=val_conf_mat)
        val_f1 = metrics.f1(confusion_matrix=val_conf_mat)
        self.metrics_dict['valid_acc'].append(val_acc)
        self.metrics_dict['valid_f1'].append(val_f1)
        _LOGGER.info("eval metrics acc, f1")
        _LOGGER.info("%s, %s", str(val_acc), str(val_f1))
        train_conf_mat = evaluate(model=model, data=loaders_dict[TRAIN],
                                  device=self.device)
        train_acc = metrics.accuracy(confusion_matrix=train_conf_mat)
        train_f1 = metrics.f1(confusion_matrix=train_conf_mat)
        self.metrics_dict['train_acc'].append(train_acc)
        self.metrics_dict['train_f1'].append(train_f1)
        _LOGGER.info("train metrics acc, f1")
        _LOGGER.info("%s, %s", str(train_acc), str(train_f1))

    def batch_train_end(self, batch_num, data, model, batch_loss):
        # super(DetailedMeasurementTrainingHook, self).batch_train_end(
        #    batch_num=batch_num, data=data, model=model, batch_loss=batch_loss)
        if batch_num % 1024 == 0 and batch_num > 0:
            val_conf_mat = evaluate(model=model, data=self.loaders_dict[VALID],
                                    device=self.device)
            val_acc = metrics.accuracy(confusion_matrix=val_conf_mat)
            val_f1 = metrics.f1(confusion_matrix=val_conf_mat)
            self.metrics_dict['valid_acc'].append(val_acc)
            self.metrics_dict['valid_f1'].append(val_f1)
            _LOGGER.info("eval metrics, batch: %s acc, f1", str(batch_num))
            _LOGGER.info("%s, %s", str(val_acc), str(val_f1))

        if batch_num == 4096:
            train_conf_mat = evaluate(model=model, data=self.loaders_dict[TRAIN],
                                      device=self.device)
            train_acc = metrics.accuracy(confusion_matrix=train_conf_mat)
            train_f1 = metrics.f1(confusion_matrix=train_conf_mat)
            self.metrics_dict['train_acc'].append(train_acc)
            self.metrics_dict['train_f1'].append(train_f1)
            _LOGGER.info("train metrics, batch: %s  acc, f1 ", str(batch_num))
            _LOGGER.info("%s, %s", str(train_acc), str(train_f1))

    def batch_valid_end(self, batch_num, data, model, batch_loss):
        pass

    def epoch_end(self, epoch, num_epochs, loaders_dict, model, train_loss, valid_loss):
        super(DetailedMeasurementTrainingHook, self).epoch_end(
            epoch=epoch, num_epochs=num_epochs, loaders_dict=loaders_dict, model=model,
            train_loss=train_loss, valid_loss=valid_loss)
        _LOGGER.info("epoch end metrics")
        self.metrics_dict['train_loss'].append(train_loss)
        self.metrics_dict['valid_loss'].append(valid_loss)
        val_conf_mat = evaluate(model=model, data=loaders_dict[VALID], device=self.device)
        val_acc = metrics.accuracy(confusion_matrix=val_conf_mat)
        val_f1 = metrics.f1(confusion_matrix=val_conf_mat)
        self.metrics_dict['valid_acc'].append(val_acc)
        self.metrics_dict['valid_f1'].append(val_f1)
        _LOGGER.info("eval metrics acc, f1 ")
        _LOGGER.info("%s, %s", str(val_acc), str(val_f1))
        train_conf_mat = evaluate(model=model, data=loaders_dict[TRAIN],
                                  device=self.device)
        train_acc = metrics.accuracy(confusion_matrix=train_conf_mat)
        train_f1 = metrics.f1(confusion_matrix=train_conf_mat)
        self.metrics_dict['train_acc'].append(train_acc)
        self.metrics_dict['train_f1'].append(train_f1)
        _LOGGER.info("train metrics acc, f1 ")
        _LOGGER.info("%s, %s", str(train_acc), str(train_f1))
