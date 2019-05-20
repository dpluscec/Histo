from sklearn import metrics
import matplotlib


try:
    matplotlib.use('Agg')
except Exception:
    pass
finally:
    import matplotlib.pyplot as plt


# Confussion matrix metrics
def accuracy(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return (tp+tn)/(tn+fp+fn+tp)


def recall(confusion_matrix):
    _, _, fn, tp = confusion_matrix.ravel()
    return tp/(tp+fn)


def specificity(confusion_matrix):
    tn, fp, _, _ = confusion_matrix.ravel()
    return tn/(tn+fp)


def precision(confusion_matrix):
    _, fp, _, tp = confusion_matrix.ravel()
    return tp/(tp+fp)


def fall_out(confusion_matrix):
    tn, fp, _, _ = confusion_matrix.ravel()
    return fp/(fp+tn)


def f1(confusion_matrix):
    _, fp, fn, tp = confusion_matrix.ravel()
    return 2*tp/(2*tp+fp+fn)


def output_metrics(confusion_matrix, conf_metrics):
    print("Evaluation results")
    print(confusion_matrix)
    print("#############################")
    for name in conf_metrics:
        print(name)
        print(conf_metrics[name](confusion_matrix))
        print("------------------------")
    print("#############################")

confusion_matrix_metrics_dict = {
    "Accuracy": accuracy,
    "Recall": recall,
    "Specificity": specificity,
    "Precision": precision,
    "Fall_out": fall_out,
    "F1": f1
    }


def plot_roc_curve(experiment_name, y_true, y_score):
    # ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', markersize=1)
    plt.savefig(f'{experiment_name}_roc.png')
