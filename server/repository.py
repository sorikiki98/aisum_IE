import importlib
from image_embedding_model import ImageEmbeddingModel
from object_detection_model import ObjectDetectionModel


class ModelRepository:
    def __init__(self, config):
        self._models = {}
        self.config = config

    def get_model_by_name(self, model_name: str):
        if model_name not in self._models:
            self._add_model_by_name(model_name)
        return self._models[model_name]

    def _add_model_by_name(self, model_name: str):
        model_cfg = self.config["model"][model_name]
        module = importlib.import_module(model_cfg["model_dir"])
        class_name = model_cfg["model_name"]
        cls = getattr(module, class_name)
        if issubclass(cls, ImageEmbeddingModel):
            model = cls(model_name, self.config)
        elif issubclass(cls, ObjectDetectionModel):
            model = cls(model_name, self.config)
        else:
            raise TypeError(f"{class_name} does not inherit from ImageEmbeddingModel or ObjectDetectionModel")
        self._models[model_name] = model

    def clear(self):
        self._models = {}
