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


def output_metrics(confusion_matrix, metrics):
    print("Evaluation results")
    print(confusion_matrix)
    print("#############################")
    for name in metrics:
        print(name)
        print(metrics[name](confusion_matrix))
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
