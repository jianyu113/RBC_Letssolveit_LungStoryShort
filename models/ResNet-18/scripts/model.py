"""ResNet-18 model setup for binary lung-opacity classification."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(pretrained=True) -> nn.Module:
    """Create an ImageNet-pretrained ResNet-18 with one output logit."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 1)

    return model
