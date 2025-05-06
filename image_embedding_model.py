from abc import ABC, abstractmethod
from torchvision import transforms
import torch.nn as nn
import torch


class ImageEmbeddingModel(nn.Module, ABC):
    def __init__(self, model_name, cfg):
        super().__init__()
        self._name = model_name
        self._preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.root_path = cfg["root_path"]
        self.model_cfg = cfg["model"][model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def forward(self, pil_images):
        pass

    @property
    def name(self):
        return self._name
