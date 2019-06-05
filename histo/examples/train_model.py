"""Module contains example for training model on PCam Dataset."""
# pylint: disable=C0103
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import histo.dataset as dataset
import histo.models as models
import histo.train as train
import histo.metrics as metrics


if __name__ == "__main__":
    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameters
    lr = 1e-5
    batch_size = 32
    num_epochs = 5
    num_outputs = 1
    weight_decay = 0

    # development dataset
    pcam_dataset = dataset.PCamDatasets(data_transforms=dataset.PCAM_DATA_TRANSFORM)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    train_iter = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
    loaders = {train.TRAIN: train_iter, train.VALID: valid_iter}

    # obtain model
    model = models.get_alexnet(
        num_outputs=num_outputs, pretrained=True, fixed_weights=False)
    model.to(device)

    # criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # train model
    model = train.train(model=model, loaders_dict=loaders, num_epochs=num_epochs,
                        optimizer=optimizer, criterion=criterion, device=device,
                        hook=train.DetailedMeasurementTrainingHook(device=device))
    torch.save(obj=model.state_dict(), f="curr_model.pt")

    # test set
    test_set = pcam_dataset.test
    test_iter = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # validation
    print("train set")
    train_cmat = train.evaluate(model=model, data=train_iter, device=device)
    metrics.output_metrics(confusion_matrix=train_cmat,
                           conf_metrics=metrics.confusion_matrix_metrics_dict)

    print("valid set")
    valid_cmat = train.evaluate(model=model, data=valid_iter, device=device)
    metrics.output_metrics(confusion_matrix=valid_cmat,
                           conf_metrics=metrics.confusion_matrix_metrics_dict)

    print("test set")
    test_cmat = train.evaluate(model=model, data=test_iter, device=device)
    metrics.output_metrics(confusion_matrix=test_cmat,
                           conf_metrics=metrics.confusion_matrix_metrics_dict)
