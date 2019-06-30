"""Module contains functions for calculating statistics on PCam dataset."""
import numpy as np
import matplotlib
import torchvision.transforms as transforms
from histo.dataset import PCamDatasets
try:
    matplotlib.use('Agg')
except ImportError:
    pass
finally:
    import matplotlib.pyplot as plt

def visualize_transformations():
    """Function creates visualization of choosen transformations for one image from PCam
    dataset."""
    pcam_ds = PCamDatasets()
    train = pcam_ds.train
    transformations = {
        "Bez transformacije": transforms.Compose([
            transforms.ToTensor(), ]),
        "Rotacija 360° uniformno": transforms.Compose([
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(), ]),
        "Rotacija 5° uniformno": transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(), ]),
        "Vertikalno zrcaljenje": transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(), ]),
        "Horizontalno zrcaljenje": transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(), ]),
        "Promjena nijanse boje": transforms.Compose([
            transforms.ColorJitter(hue=0.025),
            transforms.ToTensor(), ]),
        "Promjena zasićenja boje": transforms.Compose([
            transforms.ColorJitter(saturation=0.05),
            transforms.ToTensor(), ]),
        "Promjena svjetline boje": transforms.Compose([
            transforms.ColorJitter(brightness=0.05),
            transforms.ToTensor(), ]),
        "Promjena kontrasta boje": transforms.Compose([
            transforms.ColorJitter(contrast=0.05),
            transforms.ToTensor(), ]),
        "Kombinacija transformacija": transforms.Compose(
            transforms=[
                transforms.RandomApply(
                    [transforms.RandomOrder(transforms=[
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ColorJitter(brightness=0.05),
                        transforms.ColorJitter(contrast=0.05),
                        transforms.ColorJitter(hue=0.025),
                        transforms.RandomRotation(degrees=5)])], p=0.5),
                transforms.ToTensor()])
    }
    example = train[0][0]

    labels = []
    images = []
    for label in transformations:
        trans = transformations[label]
        tmp_img = example.copy()
        transformed = trans(tmp_img).permute(1, 2, 0).numpy()
        images.append(transformed)
        labels.append(label)
    show_multiple_images(images=images, shape=(2, 5), title="Vizualizacija "
                         "transformacija", subtitles=labels, save=True)


def save_img(fname, img):
    """Method saves given image on given path.

    Parameters
    ----------
    fname : str
        file name
    img : array like
        array containing image
    """
    plt.imsave(fname=fname, arr=img)


def stats(dataset):
    """Function outputs dataset statistics such as shape, average value and standard
    deviation.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset instance
    """
    data = dataset.data
    target = dataset.target
    print("Data shape:", data.shape)
    print("Target shape:", target.shape)
    print("Data stats:")
    print("\tAvg:", np.mean(data, axis=(0, 1, 2)))
    print("\tStd:", np.std(data, axis=(0, 1, 2)))


def pcam_stas_fun(fun):
    """Method runs given function on all PCam dataset splits.

    Parameters
    ----------
    fun : callable
        callable that accepts PyTorch dataset  instance and calculates statistics
    """
    pcam_ds = PCamDatasets()
    print("Train")
    train = pcam_ds.train
    fun(train)
    print("Valid")
    valid = pcam_ds.valid
    fun(valid)
    print("Test")
    test = pcam_ds.test
    fun(test)


def label_stats(dataset):
    """Function calculates and outputs label frequencies.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset instance
    """
    target = dataset.target
    unique, counts = np.unique(target, return_counts=True)
    freqs = dict(zip(unique, counts))
    print("Label frequencies", freqs)


def pcam_stats():
    """Function that calls statistics on PCam x data."""
    pcam_stas_fun(stats)


def pcam_label_stats():
    """Function that calls statistics on PCam target y data."""
    pcam_stas_fun(label_stats)


def show_multiple_images(images, shape, title, subtitles=None, save=False):
    """Function plots multiple images on a figure with given title.

    Parameters
    ----------
    images : array like
        images that needs to be ploted
    shape : tuple(int,int)
        tuple containing number of rows and columns for ploting images
    title : str
        plot title
    subtitles : list(str), optional
        title for every subplot
    save : bool, optional
        if true image is saved to title.png image, otherwise it is plotted
    """
    fig = plt.figure(figsize=(12, 12), dpi=300)
    fig.suptitle(title, fontsize=16)
    rows = shape[0]
    cols = shape[1]

    for i in range(1, min(cols * rows, len(images)) + 1):
        img = images[i - 1]
        curr_ax = fig.add_subplot(rows, cols, i)
        if subtitles:
            curr_ax.title.set_text(subtitles[i-1])
        plt.imshow(img)
    plt.tight_layout()
    if save:
        plt.savefig(fname=f"{title}.png")
        return
    plt.show()


def white_dataset_histogram(dataset):
    """Function calculates frequency of number of white pixels in dataset images.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    dataset instance
    """
    freqs = {}
    for ex in dataset:
        img = ex[0].numpy()
        img_sum = np.sum(img, axis=(0))
        hist, _ = np.histogram(img_sum, [i * 0.2 for i in range(16)])
        white_bin = hist[-1]
        freqs[white_bin] = freqs.get(white_bin, 0) + 1
    return freqs


def visualize_examples():
    """Function visualizes first 9 positive and 9 negative images from PCam train
    dataset."""
    pcam_ds = PCamDatasets()
    train_dataset = pcam_ds.train
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
    visualize_transformations()
