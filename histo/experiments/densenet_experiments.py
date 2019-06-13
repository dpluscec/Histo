"""Module contains DenseNet experiment definitions."""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import histo.models as models
from histo.experiments.base_experiment import (ExperimentParameters, Experiment,
                                               NUM_CLASSES)
import histo.dataset as dataset


def base_densenet_experiment(experiment_name, learn_rate, batch_size,
                             validation_batch_size, num_epochs, weight_decay, pretrained,
                             fixed_weights, data_dict, device):
    """Base densenet experiment setup.

    Parameters
    ----------
    experiment_name : str
        current experiment name
    learn_rate : float
        learning rate
    batch_size : int
        training batch size
    validation_batch_size : int
        validation batch size
    num_epochs : int
        total number of epochs used in training
    weight_decay : float
        L2 weight decay constant
    pretrained : bool
        if true pretrained model will be used
    fixed_weights : bool
        if true all but last fully connected layers will be freezed while training
    data_dict : dict(str, torch.utils.data.DataLoader)
        dictionary with training, validation and test dataloaders
    device : torch.device
        device on which to perform operations

    Returns
    -------
    experiment : Experiment
        experiment instance
    """
    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=pretrained, fixed_weights=fixed_weights)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_1(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_1", learn_rate=1e-5, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_2(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_2", learn_rate=1e-3, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_3(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_3", learn_rate=1e-4, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_4(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_4", learn_rate=1e-5, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=True, data_dict=data_dict, device=device)


def get_experiment_densenet_5(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_5", learn_rate=1e-3, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=True, data_dict=data_dict, device=device)


def get_experiment_densenet_6(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_6", learn_rate=1e-4, batch_size=32,
        validation_batch_size=1024, num_epochs=10, weight_decay=0, pretrained=True,
        fixed_weights=True, data_dict=data_dict, device=device)


def get_experiment_densenet_7(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_7", learn_rate=1e-6, batch_size=32,
        validation_batch_size=1024, num_epochs=8, weight_decay=0, pretrained=True,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_8(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_8", learn_rate=1e-6, batch_size=32,
        validation_batch_size=1024, num_epochs=8, weight_decay=0, pretrained=True,
        fixed_weights=True, data_dict=data_dict, device=device)


def get_experiment_densenet_9(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_9", learn_rate=1e-3, batch_size=32,
        validation_batch_size=1024, num_epochs=20, weight_decay=0, pretrained=False,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_10(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_10", learn_rate=1e-4, batch_size=32,
        validation_batch_size=1024, num_epochs=20, weight_decay=0, pretrained=False,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_11(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_11", learn_rate=1e-5, batch_size=32,
        validation_batch_size=1024, num_epochs=20, weight_decay=0, pretrained=False,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_12(data_dict, device):
    """See base_densenet_experiment"""
    return base_densenet_experiment(
        experiment_name="densenet_12", learn_rate=1e-6, batch_size=32,
        validation_batch_size=1024, num_epochs=20, weight_decay=0, pretrained=False,
        fixed_weights=False, data_dict=data_dict, device=device)


def get_experiment_densenet_rotation(device):
    pcam_train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_rotation"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_horizontalflip(device):
    pcam_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "horizontalflip"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_verticalflip(device):
    pcam_train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "verticalflip"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_brightness(device):
    pcam_train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.05),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_brightness"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_contrast(device):
    pcam_train_transform = transforms.Compose([
        transforms.ColorJitter(contrast=0.05),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_contrast"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_hue(device):
    pcam_train_transform = transforms.Compose([
        transforms.ColorJitter(hue=0.025),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_hue"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_saturation(device):
    pcam_train_transform = transforms.Compose([
        transforms.ColorJitter(saturation=0.05),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_saturation"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_densenet_small_rotation(device):
    pcam_train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
    ])
    pcam_valid_transform = transforms.Compose([
        transforms.ToTensor()])
    pcam_test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pcam_data_transform = {
        'train': pcam_train_transform,
        'valid': pcam_valid_transform,
        'test': pcam_test_transform
    }
    pcam_dataset = dataset.PCamDatasets(data_transforms=pcam_data_transform)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    experiment_name = "densenet_smallrotation"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 20
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_densenet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment
