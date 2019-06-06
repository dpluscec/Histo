"""Module contains example on evaluating saved model on test set."""
# pylint: disable=C0103
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
import histo.train as train
import histo.dataset as ds
import histo.metrics as hmetrics
import histo.models as models


if __name__ == "__main__":
    model_name = input("Input model name: ")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = models.get_resnet(num_outputs=1, pretrained=False)
    model.load_state_dict(torch.load(f=f"models/{model_name}.pth"))
    model.to(device)
    model.eval()

    # load data
    base_data = ds.PCamDatasets(data_transforms=ds.PCAM_DATA_TRANSFORM)
    test_set = base_data.test
    test_iter = DataLoader(dataset=test_set, batch_size=1024, shuffle=False)

    predictions, labels = train.predict_data(
        model=model, data=test_iter, device=device, return_labels=True)

    print(model_name)

    # Accuracy and F1 metric
    confusion_matrix = train.evaluate(model, test_iter, device)
    print("Accuracy:", hmetrics.accuracy(confusion_matrix=confusion_matrix))
    print("F1:", hmetrics.f1(confusion_matrix=confusion_matrix))

    hmetrics.plot_roc_curve(
        experiment_name=model_name, y_true=labels, y_score=predictions)

    print("AUC:", metrics.roc_auc_score(y_true=labels, y_score=predictions))
