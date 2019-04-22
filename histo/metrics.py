## Confussion matrix metrics

def accuracy(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return (tp+tn)/(tn+fp+fn+tp)


def recall(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tp/(tp+fn)


def specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tn/(tn+fp)


def precision(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tp/(tp+fp)


def fall_out(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return fp/(fp+tn)


def f1(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return 2*tp/(2*tp+fp+fn)