import os
import time
import logging
import torch
from torch.utils.data import DataLoader
import histo.train as train
import histo.metrics as metrics
from histo.dataset import TRAIN, VALID, TEST

_LOGGER = logging.getLogger(__name__)
MODELS_SAVE_PATH = "models"
NUM_CLASSES = 1


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
