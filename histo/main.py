import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from histo.dataset import PCamDatasets
import histo.models as models
import histo.train as train


def save_img(fname, img):
    plt.imsave(fname=fname, arr=img)


if __name__ == "__main__":
    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # hyperparameters
    lr = 1e-3
    batch_size = 256
    num_epochs = 30
    num_outputs = 2
    weight_decay = 1e-4

    # development dataset
    dataset = PCamDatasets() ## add normalization
    train_set = dataset.train
    valid_set = dataset.valid
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    loaders = {train.TRAIN: train_iter, train.VALID: valid_iter}

    # obtain model
    model = models.get_alexnet(
        num_outputs=num_outputs, pretrained=True, fixed_weights=False)
    model.to(device)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # train model
    train.train(model=model, loaders_dict=loaders, num_epochs=num_epochs,
                optimizer=optimizer, criterion=criterion, device=device)
    
    # test set
    test_set = dataset.test
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # validation
    print("train set")
    train.eval(model=model, data=train_iter, device=device)
    print("valid set")
    train.eval(model=model, data=valid_iter, device=device)
    print("test set")
    train.eval(model=model, data=test_iter, device=device)


    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
