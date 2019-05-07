import time
import copy
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import histo.metrics as metrics

TRAIN = 'train'
VALID = 'valid'


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
                           train_loss=train_loss, valid_loss=valid_loss)

    if hook_flag:
        hook.training_end()

    # load best model weights
    model.load_state_dict(best_model_wts)
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
            m = confusion_matrix(y_true, y_pred)
            result_mat += m
    return result_mat


class TrainingHook:
    def __init__(self):
        pass

    def training_start(self, model, loaders_dict, criterion):
        pass

    def training_end(self):
        pass

    def epoch_start(self, epoch, num_epochs, loaders_dict):
        pass

    def epoch_end(self, epoch, num_epochs, loaders_dict, train_loss, valid_loss):
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

    def training_end(self):
        print()
        time_elapsed = time.time()-self.train_start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def epoch_start(self, epoch, num_epochs, loaders_dict):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

    def epoch_end(self, epoch, num_epochs, loaders_dict, train_loss, valid_loss):
        print()
        print(f"Epoch loss - train: {train_loss}")
        print(f"Epoch loss - valid: {valid_loss}")

    def batch_train_end(self, batch_num, data, model, batch_loss):
        print("[Batch]: {}/{}, loss {:.5f}".format(
            batch_num + 1, len(data), batch_loss), end='\r', flush=True)

    def batch_valid_end(self, batch_num, data, model, batch_loss):
        print("[Batch]: {}/{}, loss {:.5f}".format(
            batch_num + 1, len(data), batch_loss), end='\r', flush=True)


class DetailedMeasurementTrainingHook(BasicTrainingHook):
    def __init__(self, device):
        super(DetailedMeasurementTrainingHook, self).__init__()
        self.device = device
        self.loaders_dict = None

    def training_start(self, model, loaders_dict, criterion):
        super(DetailedMeasurementTrainingHook, self).training_start(
            model=model, loaders_dict=loaders_dict, criterion=criterion)
        self.loaders_dict = loaders_dict
        print("start metrics")
        val_conf_mat = evaluate(model=model, data=loaders_dict[VALID], device=self.device)
        print("eval metrics, batch: ", metrics.accuracy(confusion_matrix=val_conf_mat),
              metrics.f1(confusion_matrix=val_conf_mat))
        train_conf_mat = evaluate(model=model, data=loaders_dict[TRAIN],
                                  device=self.device)
        print("train metrics", metrics.accuracy(confusion_matrix=train_conf_mat),
              metrics.f1(confusion_matrix=train_conf_mat))

    def batch_train_end(self, batch_num, data, model, batch_loss):
        super(DetailedMeasurementTrainingHook, self).batch_train_end(
            batch_num=batch_num, data=data, model=model, batch_loss=batch_loss)

        if batch_num % 256:
            val_conf_mat = evaluate(model=model, data=self.loaders_dict[VALID],
                                    device=self.device)
            print("eval metrics, batch: ", batch_num,
                  metrics.accuracy(confusion_matrix=val_conf_mat),
                  metrics.f1(confusion_matrix=val_conf_mat))

        if batch_num % 1024:
            train_conf_mat = evaluate(model=model, data=self.loaders_dict[TRAIN],
                                      device=self.device)
            print("train metrics", metrics.accuracy(confusion_matrix=train_conf_mat),
                  metrics.f1(confusion_matrix=train_conf_mat))
