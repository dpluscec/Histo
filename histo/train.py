import time
import copy
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import histo.metrics as metrics

TRAIN = 'train'
VALID = 'valid'


class TrainingHook:
    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def batch_start(self):
        pass

    def batch_end(self):
        pass


def train(model, loaders_dict, num_epochs, optimizer, criterion, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = None

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        run_epoch(model, loaders_dict[TRAIN], optimizer, criterion,
                  phase=TRAIN, device=device)
        run_epoch(model, loaders_dict[VALID], optimizer, criterion,
                  phase=VALID, device=device)
        train_conf_mat = evaluate(model, loaders_dict[TRAIN], device)
        print("train metrics", 
              metrics.accuracy(train_conf_mat), metrics.f1(train_conf_mat))
        eval_conf_mat = evaluate(model, loaders_dict[VALID], device)
        print("eval metrics",
              metrics.accuracy(eval_conf_mat), metrics.f1(eval_conf_mat))
    print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run_epoch(model, data, optimizer, criterion, phase, device):
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
    '''
    if phase == TRAIN:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    batch_n = 0
    for batch_num, (batch_x, batch_y) in enumerate(data):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        model.zero_grad()  # clear previous gradients

        with torch.set_grad_enabled(phase == TRAIN):
            logits = model(batch_x)
            loss = criterion(input=logits, target=batch_y)
            running_loss += loss
            if phase == TRAIN:
                loss.backward()
                optimizer.step()

        print("[Batch]: {}/{}, loss {:.5f}".format(
            batch_num + 1, len(data), loss), end='\r', flush=True)
        batch_n = batch_num
    print()
    print(f"Epoch loss: {running_loss/(batch_n+1)}")


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
