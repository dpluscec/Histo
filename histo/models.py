import torch.nn as nn
import torchvision.models as models


def get_alexnet(num_outputs, pretrained=True, fixed_weights=False):
    alexnet = models.alexnet(pretrained=pretrained)
    if fixed_weights:
        for param in alexnet.features.parameters():
            param.requires_grad = False
    num_features = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = nn.Linear(num_features, num_outputs)
    alexnet.classifier[-1].requires_grad = True
    return alexnet


def get_resnet(num_outputs, pretrained=True, fixed_weights=False):
    resnet = models.resnet18(pretrained=pretrained)
    if fixed_weights:
        for param in resnet.parameters():
            param.requires_grad = False
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_outputs)
    return resnet


def get_inception(num_outputs, pretrained=True, fixed_weights=False):
    inception = models.inception_v3(pretrained=pretrained)
    if fixed_weights:
        for param in inception.parameters():
            param.requires_grad = False
    num_features = inception.fc.in_features
    inception.fc = nn.Linear(num_features, num_outputs)
    return inception


def get_dummy_model(input_size, hidden_size, output_size):
    model = DummyModel(input_size, hidden_size, output_size)
    return model


class DummyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DummyModel, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        x = self.model(x)
        return x
