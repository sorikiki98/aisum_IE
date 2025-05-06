import json
import importlib
import sys
from PIL.Image import Image
from typing import List
from dataset import IndexDataset
from pgvector_database import PGVectorDB
from image_embedding_model import ImageEmbeddingModel
from yolo import ObjectDetectionModel


def load_image_embedding_model_from_path(model_name: str, cfg: dict):
    model_cfg = cfg["model"][model_name]
    module = importlib.import_module(model_cfg["model_dir"])
    class_name = model_cfg["model_name"]
    cls = getattr(module, class_name)

    if not issubclass(cls, ImageEmbeddingModel):
        raise TypeError(f"{class_name} does not inherit from ImageEmbeddingModel")

    return cls(model_name, cfg)


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])

    image_embedding_model_name = input("Enter embedding model name: ")

    if image_embedding_model_name not in config["model"]:
        raise ValueError("Invalid embedding model name.")

    dataset = IndexDataset("eseltree", config)
    database = PGVectorDB(image_embedding_model_name, config)
    detection_model = ObjectDetectionModel(config)

    embedding_model = load_image_embedding_model_from_path(image_embedding_model_name, config)
    batch_size = config["model"][image_embedding_model_name]["batch_size"]

    len_index_images = len(dataset.index_image_ids)
    total_batches = len_index_images // batch_size + (1 if len_index_images % batch_size > 0 else 0)
    all_embeddings = []

    for batch_idx in range(total_batches):
        batch_images, batch_ids, batch_cat1s, batch_cat2s = dataset.prepare_index_images(batch_idx, batch_size)
        batch_detection_classes, batch_detection_coordinates, batch_detection_images, batch_detection_ids = \
            detection_model(batch_images, batch_ids)
        batch_embeddings_ndarray = embedding_model(batch_detection_images)
        database.insert_image_embeddings_into_postgres(batch_detection_ids, batch_embeddings_ndarray,
                                                       batch_detection_classes, batch_detection_classes)
