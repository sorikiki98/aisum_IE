import json
import importlib
import sys
from tqdm import tqdm
from dataset import IndexDataset
from pgvector_database import PGVectorDB
from image_embedding_model import ImageEmbeddingModel
from object_detection_model import ObjectDetectionModel


def load_model_from_path(model_name: str, cfg: dict):
    model_cfg = cfg["model"][model_name]
    module = importlib.import_module(model_cfg["model_dir"])
    class_name = model_cfg["model_name"]
    cls = getattr(module, class_name)

    if issubclass(cls, ImageEmbeddingModel):
        return cls(model_name, cfg)
    elif issubclass(cls, ObjectDetectionModel):
        return cls(model_name, cfg, True)
    else:
        raise TypeError(f"{class_name} does not inherit from ImageEmbeddingModel or ObjectDetectionModel")


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])

    image_embedding_model_name = input("Enter embedding model name: ")

    if image_embedding_model_name not in config["model"]:
        raise ValueError("Invalid embedding model name.")
    # Vector_DB 카테고리별 테이블 생성
    database = PGVectorDB(image_embedding_model_name, config)
    indexed_codes = database.get_pgvector_info()["indexed_codes"]

    dataset = IndexDataset("eseltree", config)
    dataset.truncate_index_images(indexed_codes)

    detection_model = load_model_from_path("yolo", config)
    embedding_model = load_model_from_path(image_embedding_model_name, config)
    batch_size = config["model"][image_embedding_model_name]["batch_size"]

    len_index_images = len(dataset.index_image_ids)
    total_batches = len_index_images // batch_size + (1 if len_index_images % batch_size > 0 else 0)
    all_embeddings = []

    for batch_idx in tqdm(range(total_batches), desc=f"Indexing {len_index_images} Images"):
        batch_images, batch_ids = dataset.prepare_index_images(batch_idx, batch_size)
        batch_detection_result = detection_model(batch_images, batch_ids)

        batch_detection_classes = batch_detection_result["detection_classes"]
        batch_detection_images = batch_detection_result["detection_images"]
        batch_image_segment_ids = batch_detection_result["image_segment_ids"]
        batch_original_image_ids = batch_detection_result["original_image_ids"]

        if not batch_detection_images:
            print(f"Batch {batch_idx}: No detection images, skipping.")
            continue

        batch_embeddings_ndarray = embedding_model(batch_detection_images)
        # vector_DB 카테고리별 테이블 insert
        database.insert_embeddings(
            batch_image_segment_ids, batch_original_image_ids,
            batch_embeddings_ndarray, batch_detection_classes
        )
    # # vector_DB 카테고리 테이블별 인덱싱(색인 생성)
    # database.create_index()
