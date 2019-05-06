import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import histo.dataset as dataset
import histo.models as models
import histo.train as train
import histo.metrics as metrics


def save_img(fname, img):
    plt.imsave(fname=fname, arr=img)


if __name__ == "__main__":
    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameters
    lr = 1e-3
    batch_size = 32
    num_epochs = 10
    num_outputs = 1
    weight_decay = 0

    # development dataset
    pcam_dataset = dataset.PCamDatasets(data_transforms=dataset.PCAM_DATA_TRANSFORM)  # add normalization
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    loaders = {train.TRAIN: train_iter, train.VALID: valid_iter}

    # obtain model
    #model = models.get_alexnet(
    #    num_outputs=num_outputs, pretrained=True, fixed_weights=True)
    #model = models.get_dummy_model(input_size=96*96*3, hidden_size=256, output_size=1)
    model = models.get_resnet(num_outputs=num_outputs)
    model.to(device)
    # criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    #optimizer = torch.optim.Adam(
     #   params=model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    # train model
    train.train(model=model, loaders_dict=loaders, num_epochs=num_epochs,
                optimizer=optimizer, criterion=criterion, device=device)

    # test set
    test_set = pcam_dataset.test
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # validation
    print("train set")
    train_cmat = train.evaluate(model=model, data=train_iter, device=device)
    metrics.output_metrics(train_cmat, metrics.confusion_matrix_metrics_dict)
    print("valid set")
    valid_cmat = train.evaluate(model=model, data=valid_iter, device=device)
    metrics.output_metrics(valid_cmat, metrics.confusion_matrix_metrics_dict)
    print("test set")
    test_cmat = train.evaluate(model=model, data=test_iter, device=device)
    metrics.output_metrics(test_cmat, metrics.confusion_matrix_metrics_dict)

    '''
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    '''
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
