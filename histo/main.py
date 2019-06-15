"""Main module containing code for running experiments. Note if you want to use inception
you should add aditional transformation to dataset - resize images to 299x299."""
# pylint: disable=C0103
import random
import logging
import numpy as np
import torch

import histo.dataset as dataset
import histo.experiments as experiments


_LOGGER = logging.getLogger(__name__)


def set_random_seed():
    """Function initializes random seed in python, numpy and pytorch."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


if __name__ == "__main__":
    set_random_seed()

    # constants
    num_classes = 1

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _LOGGER.info(device)

    # development dataset
    pcam_dataset = dataset.PCamDatasets(data_transforms=dataset.PCAM_DATA_TRANSFORM)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    # experiment
    series_one = [
        experiments.get_experiment_densenet_1(data_dict=data_dict, device=device)
    ]
    for exp in series_one:
        exp.execute()
