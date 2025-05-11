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


class DatabaseRepository:
    def __init__(self, config):
        self._databases = {}
        self.config = config

    def get_db_by_name(self, model_name: str):
        if model_name not in self._databases:
            self._add_db_by_name(model_name)
        return self._databases[model_name]

    def _add_db_by_name(self, model_name: str):
        self._databases[model_name] = PGVectorDB(model_name, self.config)

    def clear(self):
        self._databases = {}


class ImageRetrievalRepository:
    def __init__(self, config):
        self._retrieval_results = {}
        self._model_repository = ModelRepository(config)
        self._database_repository = DatabaseRepository(config)
        self.config = config

    def get_retrieval_result_by_name(self, model_name, query_image, query_id, category):
        if model_name in self._retrieval_results:
            return self._retrieval_results[model_name]
        embedding_model = self._model_repository.get_model_by_name(model_name)
        database = self._database_repository.get_db_by_name(model_name)
        retrieval_model = ImageRetrieval(embedding_model, database, self.config)
        self._retrieval_results[model_name] = retrieval_model(query_image, query_id, category)
        return self._retrieval_results[model_name]

    def ensemble(self, query_image, query_id, category):
        ensemble_model_names = self.config["ensemble"].values()
        retrieval_results = dict()
        for name in ensemble_model_names:
            result = self.get_retrieval_result_by_name(name, query_image, query_id, category)
            retrieval_results[name] = result
        ensemble = Ensemble(retrieval_results, self.config)
        ensemble_result = ensemble()

        return ensemble_result

    def reset(self):
        self._model_repository.clear()
        self._database_repository.clear()
        self.clear_retrieval_results()

    def clear_retrieval_results(self):
        self._retrieval_results = {}
