import torch
from torch.utils.data import DataLoader
import histo.train as train
import histo.metrics as metrics


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


class ExperimentParameters:
    def __init__(self, lr=1e-5, batch_size=32, num_epochs=5, weight_decay=0):
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay


class Experiment:
    def __init__(self, name, params, data_dict, model, optimizer, criterion, device):
        self.name = name
        self.device = device
        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_dict = data_dict
        train_iter = DataLoader(dataset=self.data_dict[TRAIN], batch_size=batch_size,
                                shuffle=True)
        valid_iter = DataLoader(dataset=self.data_dict[VALID], batch_size=batch_size,
                                shuffle=False)
        test_iter = DataLoader(dataset=self.data_dict[TEST], batch_size=batch_size,
                                shuffle=False)
        self.loaders = {TRAIN: train_iter, VALID: valid_iter, TEST: test_iter}

    def execute(self):
        self._train_model()
        self._save_model()
        self._validate_experiment()

    def _train_model(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.model = train.train(
            model=self.model, loaders_dict=self.data_dict, 
            num_epochs=self.params.num_epochs, optimizer=self.optimizer,
            criterion=self.criterion, device=self.device,
            hook=train.DetailedMeasurementTrainingHook(device=self.device))

    def _validate_experiment(self):
        # validation
        print("train set")
        train_cmat = train.evaluate(model=self.model, data=self.data_dict[TRAIN],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=train_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

        print("valid set")
        valid_cmat = train.evaluate(model=self.model, data=self.data_dict[VALID],
                                    device=self.device)
        metrics.output_metrics(confusion_matrix=valid_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

        print("test set")
        test_cmat = train.evaluate(model=self.model, data=self.data_dict[TEST],
                                   device=self.device)
        metrics.output_metrics(confusion_matrix=test_cmat,
                               metrics=metrics.confusion_matrix_metrics_dict)

    def _save_model(self):
        torch.save(obj=self.model.state_dict(), f="curr_model.pt")
