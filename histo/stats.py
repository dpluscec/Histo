import numpy as np
from histo.dataset import PCamDatasets

def stats(dataset):
    data = dataset.data
    target = dataset.target
    print("Data shape:", data.shape)
    print("Target shape:", target.shape)
    print("Data stats:")
    print("\tAvg:", np.mean(data, axis=(0, 1, 2)))
    print("\tStd:", np.std(data, axis=(0, 1, 2)))


def pcam_stas_fun(fun):
    ds = PCamDatasets()
    print("Train")
    train = ds.train
    fun(train)
    print("Valid")
    valid = ds.valid
    fun(valid)
    print("Test")
    test = ds.test
    fun(test)


def label_stats(dataset):
    target = dataset.target
    unique, counts = np.unique(target, return_counts=True)
    freqs = dict(zip(unique, counts))
    print("Label frequencies", freqs)


def pcam_stats():
    pcam_stas_fun(stats)


def pcam_label_stats():
    pcam_stas_fun(label_stats)

if __name__ == "__main__":
    pcam_label_stats()    