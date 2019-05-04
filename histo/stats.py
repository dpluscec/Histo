import numpy as np
import matplotlib.pyplot as plt
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


def show_multiple_images(images, shape, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    rows = shape[0]
    cols = shape[1]

    for i in range(1, cols*rows+1):
        img = images[i-1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
    plt.show()


def visualize_examples():
    ds = PCamDatasets()
    train_dataset = ds.train
    pos_examples = []
    neg_examples = []

    index = 0
    while len(pos_examples) < 9 or len(neg_examples) < 9:
        example = train_dataset[index]
        data = example[0].permute(1, 2, 0)
        label = example[1]
        if label == 1 and len(pos_examples) < 9:
            pos_examples.append(data)
        elif label == 0 and len(neg_examples) < 9:
            neg_examples.append(data)
        index += 1
    show_multiple_images(images=pos_examples, shape=(3, 3), title="Pozitivni primjeri")
    show_multiple_images(images=neg_examples, shape=(3, 3), title="Negativni primjeri")


if __name__ == "__main__":
    visualize_examples()
