import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights


class ResNext101(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = resnext101_64x4d(ResNeXt101_64X4D_Weights)
        self._avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        layer1_out = self._model.layer1(x)  # bs, 256, 56, 56
        layer2_out = self._model.layer2(layer1_out)
        layer3_out = self._model.layer3(layer2_out)  # bs, 1024, 14, 14
        layer4_out = self._model.layer4(layer3_out)  # bs, 2048, 7, 7

        x = self._avgpool(layer4_out)  # bs, 2048, 1, 1
        bs = x.size()[0]

        x = x.view(bs, -1)
        normalized_x = F.normalize(x, dim=1)

        return normalized_x

