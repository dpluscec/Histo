# pylint: disable=C0103
import numpy as np
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import histo.train as train
import histo.dataset as ds
import histo.metrics as hmetrics
import histo.models as models


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    matplotlib.use('Agg')

    # load model
    model = models.get_alexnet(num_outputs=1, pretrained=False)
    model.load_state_dict(torch.load("models/resnet_1-1557378617.pth"))
    model.to(device)
    model.eval()

    # load data
    base_data = ds.PCamDatasets(data_transforms=ds.PCAM_DATA_TRANSFORM)
    test_set = base_data.test
    test_iter = DataLoader(dataset=test_set, batch_size=1024, shuffle=False)

    labels = []
    predictions = []

    with torch.no_grad():
        prob_output = nn.Sigmoid()
        for batch_x, batch_y in test_iter:
            batch_x = batch_x.to(device)
            labels.extend(list(batch_y.numpy()))
            logits = model(batch_x)
            probs = prob_output(logits)
            predictions.extend(list(probs.cpu().numpy()))

    labels = np.array(labels)
    predictions = np.array(predictions)

    # Accuracy and F1 metric
    confusion_matrix = train.evaluate(model, test_iter, device)
    print("Accuracy:", hmetrics.accuracy(confusion_matrix=confusion_matrix))
    print("F1:", hmetrics.f1(confusion_matrix=confusion_matrix))

    # ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=predictions)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', markersize=1)
    plt.savefig('roc_example.png')

    print("AUC:", metrics.roc_auc_score(y_true=labels, y_score=predictions))
