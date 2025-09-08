import json
import importlib
import sys
from tqdm import tqdm
from dataset import IndexDataset
from pgvector_database import PGVectorDB
from image_embedding_model import ImageEmbeddingModel
from object_detection_model import ObjectDetectionModel
import os
import mysql.connector
import numpy as np

# 임베딩 실패 시 DB 상태를 'd'로 변경하는 함수
# -------------------------------------------------------------------
def update_image_status(image_ids: list, status: str, db_cfg: dict):
    if not image_ids:
        return
    try:
        conn = mysql.connector.connect(**db_cfg["mysql"])
        cursor = conn.cursor()
        update_query = f"UPDATE crawling_list SET img_dn = %s WHERE p_key IN ({','.join(['%s'] * len(image_ids))})"
        params = [status] + image_ids
        cursor.execute(update_query, params)
        conn.commit()
    except Exception as e:
        # print문은 요청에 따라 제거
        pass
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
# -------------------------------------------------------------------

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

    if len(sys.argv) < 3:
        print("실행 오류: python index_builder.py [작업파일.txt] [모델이름]")
        print("예시: python index_builder.py ./tasks/testtask.txt dreamsim")
        sys.exit(1)
    
    task_filepath = sys.argv[1]
    image_embedding_model_name = sys.argv[2]
    # image_embedding_model_name = input("Enter embedding model name: ") # 기존 방식 제거

    if image_embedding_model_name not in config["model"]:
        raise ValueError("Invalid embedding model name.")
    
    output_dir = "embedding_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"INFO: 임베딩 결과는 '{output_dir}' 폴더에 저장됩니다.")

    database = PGVectorDB(image_embedding_model_name, config)
    indexed_codes = database.get_pgvector_info()["indexed_codes"]

    dataset = IndexDataset(task_filepath, config)
    dataset.filter_by_status(required_status=2, required_img_dn='E')
    dataset.truncate_index_images(indexed_codes)

    detection_model = load_model_from_path("yolo", config)
    embedding_model = load_model_from_path(image_embedding_model_name, config)
    batch_size = config["model"][image_embedding_model_name]["batch_size"]

    len_index_images = len(dataset.index_image_ids)
    total_batches = len_index_images // batch_size + (1 if len_index_images % batch_size > 0 else 0)
    all_embeddings = []

    for batch_idx in tqdm(range(total_batches), desc=f"Indexing {len_index_images} Images"):

        batch_images, batch_ids = dataset.prepare_index_images(batch_idx, batch_size)

        batch_detection_result   = detection_model(batch_images, batch_ids)
        batch_detection_classes  = batch_detection_result["detection_classes"]
        batch_detection_images   = batch_detection_result["detection_images"]
        batch_image_segment_ids  = batch_detection_result["image_segment_ids"]
        batch_original_image_ids = batch_detection_result["original_image_ids"]

        if not batch_detection_images:
            print(f"Batch {batch_idx}: No detection images, skipping.")
            update_image_status(batch_original_image_ids, "d", config["database"])
            continue


        batch_embeddings_ndarray = embedding_model(batch_detection_images)
            
        if batch_embeddings_ndarray is None or batch_embeddings_ndarray.size == 0:
            print(f"Batch {batch_idx}: Empty embeddings, skipping.")
            update_image_status(batch_original_image_ids, "d", config["database"])
            continue

        print(f"DEBUG: Batch {batch_idx} - Embedding array shape: {batch_embeddings_ndarray.shape}")

        # [주석 처리] 👈 6. PGVectorDB에 저장하는 부분 비활성화
        # database.insert_image_embeddings_into_postgres(
        #      batch_image_segment_ids, batch_original_image_ids,
        #      batch_embeddings_ndarray, batch_detection_classes
        #  )
        
