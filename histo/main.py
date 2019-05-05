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
    # device = torch.device("cpu")
    print(device)

    # hyperparameters
    lr = 1e-3
    batch_size = 32
    num_epochs = 10
    num_outputs = 2
    weight_decay = 0

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
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # train model
    train.train(model=model, loaders_dict=loaders, num_epochs=num_epochs,
                optimizer=optimizer, criterion=criterion, device=device)

    # test set
    test_set = dataset.test
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # validation
    print("train set")
    train_cmat = train.eval(model=model, data=train_iter, device=device)
    print("valid set")
    valid_cmat = train.eval(model=model, data=valid_iter, device=device)
    print("test set")
    test_cmat = train.eval(model=model, data=test_iter, device=device)



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
