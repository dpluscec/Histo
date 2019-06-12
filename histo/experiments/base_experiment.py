"""Module contains basic experiment logic and class for experiment parameters."""
import os
import time
import logging
from sklearn import metrics as smetrics
import torch
from torch.utils.data import DataLoader
import histo.train as train
import histo.metrics as metrics
from histo.dataset import TRAIN, VALID, TEST

_LOGGER = logging.getLogger(__name__)
MODELS_SAVE_PATH = "models"
NUM_CLASSES = 1


class ExperimentParameters:
    """Class is a holder for experiment parameters"""
    def __init__(self, lr=1e-5, batch_size=32, validation_batch_size=32,
                 num_epochs=5, weight_decay=0):
        """Constructor that initializes experiment parameters.

        Parameters
        ----------
        lr : float
            model learning rate
        batch_size : int
            batch size used when iterating over training set
        validation_batch_size : int
            batch size used when iterating over validation set
        num_epochs : int
            number of epochs for training a model
        weight_decay : float
            L2 weight decay coefficient
        """
        self.learn_rate = lr
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

    def __str__(self):
        return f"Params(lr: {self.learn_rate}, weight_decay: {self.weight_decay}, "\
               f"batch_size: {self.batch_size}, num_epochs: {self.num_epochs})"


class Experiment:
    """Class defines basic experiment for training and validating a deep learning model
    defined in PyTorch."""
    def __init__(self, name, params, data_dict, model, optimizer, criterion, device):
        """Constructor that initializes experiment. Experiment is started by using
        execute method.

        Parameters
        ----------
        name : str
            experiment name
        params : ExperimentParameters
            parameters for model training
        data_dict : dict(str, torch.utils.data.DataLoader)
            dictionary that maps dataset subset names (TRAIN, VALID, TEST) to dataloaders
        model : nn.Module
            PyTorch model used in experiment
        optimizer : torch.optim.Optimizer
            model optimizer, None if validation phase
        criterion : loss
            pytorch loss function
        device : torch.device
            device on which to perform operations
        """
        self.name = f"{name}-{str(int(time.time()))}"
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
        """Method executes the experiment."""
        _LOGGER.info("-" * 40)
        _LOGGER.info("Starting experiment %s", self.name)
        _LOGGER.info("Experiment parameters %s", str(self))
        self._train_model()
        self._save_model()
        self._validate_experiment()
        _LOGGER.info("Experiment end")
        _LOGGER.info("#" * 40)

    def _train_model(self):
        """Method starts model training."""
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.model = train.train(
            model=self.model, loaders_dict=self.loaders,
            num_epochs=self.params.num_epochs, optimizer=self.optimizer,
            criterion=self.criterion, device=self.device,
            hook=train.DetailedMeasurementTrainingHook(device=self.device))

    def _validate_experiment(self):
        """Method starts model validation on train, validation and test set"""
        # validation
        _LOGGER.info("experiment validation")
        _LOGGER.info("train set")
        train_cmat = train.evaluate(model=self.model, data=self.loaders[TRAIN],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=train_cmat,
                               conf_metrics=metrics.confusion_matrix_metrics_dict)

        _LOGGER.info("valid set")
        valid_cmat = train.evaluate(model=self.model, data=self.loaders[VALID],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=valid_cmat,
                               conf_metrics=metrics.confusion_matrix_metrics_dict)

        _LOGGER.info("test set")
        test_cmat = train.evaluate(model=self.model, data=self.loaders[TEST],
                                   device=self.device)
        metrics.output_metrics(confusion_matrix=test_cmat,
                               conf_metrics=metrics.confusion_matrix_metrics_dict)
        test_pred, test_label = train.predict_data(
            model=self.model, data=self.loaders[TEST], device=self.device,
            return_labels=True)
        _LOGGER.info(
            "AUC: %s", str(smetrics.roc_auc_score(y_true=test_label, y_score=test_pred)))
        metrics.plot_roc_curve(experiment_name=self.name,
                               y_true=test_label, y_score=test_pred)

    def _save_model(self):
        """Method saves trained model to experiment_name.pth file."""
        file_path = os.path.join(
            ".", MODELS_SAVE_PATH, f"{self.name}.pth")
        _LOGGER.info("Model saved, path %s", str(file_path))
        torch.save(obj=self.model.state_dict(), f=file_path)
