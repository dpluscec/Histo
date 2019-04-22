import torch.nn as nn
import torchvision.models as models


def get_alexnet(num_outputs, pretrained=True, fixed_weights=False):
    alexnet = models.alexnet(pretrained=pretrained)
    if fixed_weights:
        for param in alexnet.parameters():
            param.requires_grad = False
    num_features = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = nn.Linear(num_features, num_outputs)
    return alexnet


def get_resnet(num_outputs, pretrained=True, fixed_weights=False):
    resnet = models.resnet18(pretrained=pretrained)
    if fixed_weights:
        for param in resnet.parameters():
            param.requires_grad = False
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_outputs)
    return resnet
