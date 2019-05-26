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
    data_dict = {dataset.TRAIN: train_set, dataset.VALID: valid_set,
                 dataset.TEST: test_set}

    # experiment
    series_one = [
        experiments.get_experiment_densenet_1(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_2(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_3(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_4(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_5(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_6(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_7(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_8(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_9(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_10(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_11(data_dict=data_dict, device=device),
        experiments.get_experiment_densenet_12(data_dict=data_dict, device=device),
    ]
    for exp in series_one:
        exp.execute()
