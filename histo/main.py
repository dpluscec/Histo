# pylint: disable=C0103
import random
import numpy as np
import torch

import histo.dataset as dataset
import histo.experiments as experiments


if __name__ == "__main__":
    # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # constants
    num_classes = 1

    # determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # development dataset
    pcam_dataset = dataset.PCamDatasets(data_transforms=dataset.PCAM_DATA_TRANSFORM)
    train_set = pcam_dataset.train
    valid_set = pcam_dataset.valid
    test_set = pcam_dataset.test
    data_dict = {experiments.TRAIN: train_set, experiments.VALID: valid_set,
                 experiments.TEST: test_set}

    # experiment
    series_one = [
        experiments.get_experiment_alexnet_7(data_dict=data_dict, device=device),
        experiments.get_experiment_alexnet_8(data_dict=data_dict, device=device),
        experiments.get_experiment_alexnet_9(data_dict=data_dict, device=device),
        experiments.get_experiment_alexnet_10(data_dict=data_dict, device=device),
        experiments.get_experiment_resnet_7(data_dict=data_dict, device=device),
        experiments.get_experiment_resnet_8(data_dict=data_dict, device=device),
        experiments.get_experiment_resnet_9(data_dict=data_dict, device=device),
        experiments.get_experiment_resnet_10(data_dict=data_dict, device=device)]
    for exp in series_one:
        exp.execute()
