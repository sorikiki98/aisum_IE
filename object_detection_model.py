from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class ObjectDetectionModel(nn.Module, ABC):
    def __init__(self, model_name, cfg, indexing=False):
        super().__init__()
        self._name = model_name
        self.model_cfg = cfg["model"][model_name]
        self._indexing = indexing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
