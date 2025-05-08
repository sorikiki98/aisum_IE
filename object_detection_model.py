from abc import ABC, abstractmethod
import torch.nn as nn


class ObjectDetectionModel(nn.Module, ABC):
    def __init__(self, model_name, cfg):
        super().__init__()
        self._name = model_name
        self.model_cfg = cfg["model"][model_name]

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def forward(self, pil_images, img_ids):
        pass

    @property
    def name(self):
        return self._name
