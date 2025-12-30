import torch.nn as nn

from torchvision.models import AlexNet, VGG, mobilenet_v3_small

MODELS = [
    "alexnet",
]


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "alexnet":
        return mobilenet_v3_small(num_classes=num_classes)
    raise ValueError(f"Model {name} not supported. Choose from {MODELS}")
