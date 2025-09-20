import importlib
from image_embedding_model import ImageEmbeddingModel
from object_detection_model import ObjectDetectionModel
from pgvector_database import PGVectorDB
from image_retrieval import ImageRetrieval
from ensemble_retrieval import Ensemble


class ModelRepository:
    def __init__(self, config):
        self._models = {}
        self.config = config

    def get_yolo_version(self):
        return self.config["model"]["yolo"]["version"]

    def get_model_by_name(self, model_name: str):
        yolo_version = self.get_yolo_version()
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        if model_name_with_yolo_version not in self._models:
            self._add_model_by_name(model_name)
        return self._models[model_name_with_yolo_version]

    def _add_model_by_name(self, model_name: str):
        model_cfg = self.config["model"][model_name]
        yolo_version = self.get_yolo_version()
        module = importlib.import_module(model_cfg["model_dir"])
        class_name = model_cfg["model_name"]
        cls = getattr(module, class_name)
        if issubclass(cls, ImageEmbeddingModel):
            model = cls(model_name, self.config)
        elif issubclass(cls, ObjectDetectionModel):
            model = cls(model_name, self.config)
        else:
            raise TypeError(f"{class_name} does not inherit from ImageEmbeddingModel or ObjectDetectionModel")
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        self._models[model_name_with_yolo_version] = model

    def clear(self):
        self._models = {}


class DatabaseRepository:
    def __init__(self, config):
        self.databases = {}
        self.config = config

    def get_yolo_version(self):
        return self.config["model"]["yolo"]["version"]

    def get_db_by_name(self, model_name: str):
        yolo_version = self.get_yolo_version()
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        if model_name_with_yolo_version not in self.databases:
            self._add_db_by_name(model_name)
        return self.databases[model_name_with_yolo_version]

    def _add_db_by_name(self, model_name: str):
        yolo_version = self.get_yolo_version()
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        self.databases[model_name_with_yolo_version] = PGVectorDB(model_name, self.config)

    def clear(self):
        self.databases = {}


class ImageRetrievalRepository:
    def __init__(self, config):
        self._retrieval_results = {}
        self.models = ModelRepository(config)
        self.databases = DatabaseRepository(config)
        self.config = config

    def get_yolo_version(self):
        return self.config["model"]["yolo"]["version"]

    def get_retrieval_result_by_name(self, model_name, query_images, query_ids, categories, 
                                     bbox_sizes=None, bbox_centralities=None):
        # bbox_sizes, bbox_centralities 추가: p_score update에 사용
        yolo_version = self.get_yolo_version()
        model_name_with_yolo_version = f"{model_name}_yolo{yolo_version}"
        if model_name_with_yolo_version in self._retrieval_results:
            return self._retrieval_results[model_name_with_yolo_version]
        embedding_model = self.models.get_model_by_name(model_name)
        database = self.databases.get_db_by_name(model_name)
        retrieval_model = ImageRetrieval(embedding_model, database, self.config)
        self._retrieval_results[model_name_with_yolo_version] = retrieval_model(query_images, query_ids, categories, bbox_sizes, bbox_centralities)
        return self._retrieval_results[model_name_with_yolo_version]

    def ensemble(self, query_images, query_ids, categories,
                 bbox_sizes=None, bbox_centralities=None):
        ensemble_model_names = self.config["ensemble"].values()
        retrieval_results = dict()
        for name in ensemble_model_names:
            result = self.get_retrieval_result_by_name(name, query_images, query_ids, categories, bbox_sizes, bbox_centralities)
            retrieval_results[name] = result
        ensemble = Ensemble(retrieval_results, self.config)
        ensemble_result = ensemble()

        return ensemble_result

    def reset(self):
        self.models.clear()
        self.databases.clear()
        self.clear_retrieval_results()

    def clear_retrieval_results(self):
        self._retrieval_results = {}
