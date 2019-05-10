import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import histo.train as train
import histo.metrics as metrics
import histo.models as models
from histo.dataset import TRAIN, VALID, TEST

_LOGGER = logging.getLogger(__name__)
MODELS_SAVE_PATH = "models"


class ExperimentParameters:
    def __init__(self, lr=1e-5, batch_size=32, validation_batch_size=32,
                 num_epochs=5, weight_decay=0):
        self.learn_rate = lr
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

    def __str__(self):
        return f"Params(lr: {self.learn_rate}, weight_decay: {self.weight_decay}, "\
               f"batch_size: {self.batch_size}, num_epochs: {self.num_epochs})"


class Experiment:
    def __init__(self, name, params, data_dict, model, optimizer, criterion, device):
        self.name = name
        self.device = device
        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_dict = data_dict
        train_iter = DataLoader(dataset=self.data_dict[TRAIN],
                                batch_size=params.batch_size,
                                shuffle=True)
        valid_iter = DataLoader(dataset=self.data_dict[VALID],
                                batch_size=params.validation_batch_size,
                                shuffle=False)
        test_iter = DataLoader(dataset=self.data_dict[TEST],
                               batch_size=params.validation_batch_size,
                               shuffle=False)
        self.loaders = {TRAIN: train_iter, VALID: valid_iter, TEST: test_iter}

    def __str__(self):
        return f"Experiment[name: {self.name}, model: {str(self.model)}, "\
               f"params: {str(self.params)}, optimizer: {str(self.optimizer)}, "\
               f"criterion: {str(self.criterion)}]"

    def execute(self):
        _LOGGER.info("-"*40)
        _LOGGER.info("Starting experiment %s", self.name)
        _LOGGER.info("Experiment parameters %s", str(self))
        self._train_model()
        self._save_model()
        self._validate_experiment()
        _LOGGER.info("Experiment end")
        _LOGGER.info("#"*40)

    def _train_model(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.model = train.train(
            model=self.model, loaders_dict=self.loaders,
            num_epochs=self.params.num_epochs, optimizer=self.optimizer,
            criterion=self.criterion, device=self.device,
            hook=train.DetailedMeasurementTrainingHook(device=self.device))

    def _validate_experiment(self):
        # validation
        print("train set")
        train_cmat = train.evaluate(model=self.model, data=self.loaders[TRAIN],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=train_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

        print("valid set")
        valid_cmat = train.evaluate(model=self.model, data=self.loaders[VALID],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=valid_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

        print("test set")
        test_cmat = train.evaluate(model=self.model, data=self.loaders[TEST],
                                   device=self.device)
        metrics.output_metrics(confusion_matrix=test_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

    def _save_model(self):
        file_path = os.path.join(
            ".", MODELS_SAVE_PATH, f"{self.name}-{str(int(time.time()))}.pth")
        torch.save(obj=self.model.state_dict(), f=file_path)


NUM_CLASSES = 1


def get_experiment_alexnet_1(data_dict, device):
    experiment_name = "alexnet_1"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_2(data_dict, device):
    experiment_name = "alexnet_2"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_3(data_dict, device):
    experiment_name = "alexnet_3"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_4(data_dict, device):
    experiment_name = "alexnet_4"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_5(data_dict, device):
    experiment_name = "alexnet_5"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_6(data_dict, device):
    experiment_name = "alexnet_6"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_7(data_dict, device):
    experiment_name = "alexnet_7"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_8(data_dict, device):
    experiment_name = "alexnet_8"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_9(data_dict, device):
    experiment_name = "alexnet_9"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_alexnet_10(data_dict, device):
    experiment_name = "alexnet_10"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_alexnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_1(data_dict, device):
    experiment_name = "resnet_1"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_2(data_dict, device):
    experiment_name = "resnet_2"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_3(data_dict, device):
    experiment_name = "resnet_3"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_4(data_dict, device):
    experiment_name = "resnet_4"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_5(data_dict, device):
    experiment_name = "resnet_5"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_6(data_dict, device):
    experiment_name = "resnet_6"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_7(data_dict, device):
    experiment_name = "resnet_7"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_8(data_dict, device):
    experiment_name = "resnet_8"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_9(data_dict, device):
    experiment_name = "resnet_9"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_resnet_10(data_dict, device):
    experiment_name = "resnet_10"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_resnet(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_1(data_dict, device):
    experiment_name = "inception_1"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_2(data_dict, device):
    experiment_name = "inception_2"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_3(data_dict, device):
    experiment_name = "inception_3"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_4(data_dict, device):
    experiment_name = "inception_4"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_5(data_dict, device):
    experiment_name = "inception_5"
    learn_rate = 1e-3
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_6(data_dict, device):
    experiment_name = "inception_6"
    learn_rate = 1e-4
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 10
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_7(data_dict, device):
    experiment_name = "inception_7"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_8(data_dict, device):
    experiment_name = "inception_8"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_9(data_dict, device):
    experiment_name = "inception_9"
    learn_rate = 1e-5
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment


def get_experiment_inception_10(data_dict, device):
    experiment_name = "inception_10"
    learn_rate = 1e-6
    batch_size = 32
    validation_batch_size = 1024
    num_epochs = 8
    weight_decay = 0

    params = ExperimentParameters(lr=learn_rate, batch_size=batch_size,
                                  validation_batch_size=validation_batch_size,
                                  num_epochs=num_epochs, weight_decay=weight_decay)
    model = models.get_inception(
        num_outputs=NUM_CLASSES, pretrained=True, fixed_weights=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params.learn_rate, weight_decay=params.weight_decay)
    experiment = Experiment(name=experiment_name, params=params, data_dict=data_dict,
                            optimizer=optimizer, criterion=criterion,
                            device=device, model=model)
    return experiment
