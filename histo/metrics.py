"""Module contains functions for calculating different model metrics."""
# pylint: disable=C0103
import logging
from sklearn import metrics
import matplotlib

try:
    matplotlib.use('Agg')
except ImportError:
    pass
finally:
    import matplotlib.pyplot as plt


_LOGGER = logging.getLogger(__name__)


# Confussion matrix metrics
def accuracy(confusion_matrix):
    """Function calculates model accuracy based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    accuracy : float
        model accuracy
    """
    tn, fp, fn, tp = confusion_matrix.ravel()
    return (tp + tn) / (tn + fp + fn + tp)


def recall(confusion_matrix):
    """Function calculates model recall based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    recall : float
        model recall
    """
    _, _, fn, tp = confusion_matrix.ravel()
    return tp / (tp + fn)


def specificity(confusion_matrix):
    """Function calculates model specificity based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    specificity : float
        model specificity
    """
    tn, fp, _, _ = confusion_matrix.ravel()
    return tn / (tn + fp)


def precision(confusion_matrix):
    """Function calculates model precision based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    precision : float
        model precision
    """
    _, fp, _, tp = confusion_matrix.ravel()
    return tp / (tp + fp)


def fall_out(confusion_matrix):
    """Function calculates model fall_out based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    fall_out : float
        model fall_out
    """
    tn, fp, _, _ = confusion_matrix.ravel()
    return fp / (fp + tn)


def f1(confusion_matrix):
    """Function calculates model f1 based on given confussion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]

    Returns
    -------
    f1 : float
        model f1
    """
    _, fp, fn, tp = confusion_matrix.ravel()
    return 2 * tp / (2 * tp + fp + fn)


def output_metrics(confusion_matrix, conf_metrics):
    """Method logs given metrics by using confusion matrix.

    Parameters
    ----------
    confusion_matrix : array like
        2D array containing confusion matrix in form [[tp, fp],[fn, tp]]
    conf_metrics : dict(str, callable)
        dictionary that maps metrics name to function that can calculate metric based on
        confusion matrix
    """
    _LOGGER.info("Evaluation results")
    _LOGGER.info(confusion_matrix)
    _LOGGER.info("#############################")
    for name in conf_metrics:
        _LOGGER.info(name)
        _LOGGER.info(conf_metrics[name](confusion_matrix))
        _LOGGER.info("------------------------")
    _LOGGER.info("#############################")


confusion_matrix_metrics_dict = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Specificity": specificity,
    "Precision": precision,
    "Fall_out": fall_out,
    "F1": f1
}


def plot_roc_curve(experiment_name, y_true, y_score):
    """Method plots roc curve and saves it to experiment_name_roc.png file.

    Parameters
    ----------
    experiment_name : str
        experiment name used in file name
    y_true : array like
        array containing true predictions
    y_score : array like
        array containing model predictions
    """
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', markersize=1)
    plt.savefig(f'{experiment_name}_roc.png')
    plt.clf()
