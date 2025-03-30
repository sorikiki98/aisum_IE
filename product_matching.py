import os
import shutil
from PIL import Image
from pathlib import Path
from image_embedding_model import *
from vector_database import *
from pgvector_database import *


def find_similar_product_ids(image_embedding_model,
                             image_embedding_model_name,
                             faiss_index_with_ids,
                             model_params=None):
    query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, similarities = search_faiss_index(faiss_index_with_ids, query_embeddings, 3, 1024)
    return ids, similarities


def save_retrieved_images_by_ids(image_embedding_model_name, all_batch_ids, all_batch_similarities, category1, category2):
    # ✅ WSL 경로에 맞게 프로젝트 루트 고정
    project_root = Path("/mnt/c/Users/SMU/Documents/aisum_IE")
    data_root = project_root / "data"

    for i, batch_ids in enumerate(all_batch_ids):
        # 분석용 outputs 폴더
        retrieved_image_folder = project_root / "outputs" / image_embedding_model_name / str(i)
        if retrieved_image_folder.exists():
            shutil.rmtree(retrieved_image_folder)
        retrieved_image_folder.mkdir(parents=True, exist_ok=True)

        for j, img_id in enumerate(batch_ids):
            original_path = data_root / category1 / category2 / f"{img_id}.jpg"
            public_image_path = data_root / category1 / category2 / f"{img_id}.jpg"

            if original_path.exists():
                image = Image.open(original_path)
                save_path = retrieved_image_folder / f"top_{j + 1}_{all_batch_similarities[i][j]}.jpg"
                image.save(save_path)

                # ✅ 같은 파일이 아닐 때만 복사
                if original_path != public_image_path:
                    public_image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(original_path), str(public_image_path))

                print(f"✅ 복사 완료: {public_image_path}")
            else:
                print(f"❌ 원본 이미지 없음: {original_path}")


def find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, model_params=None):
    query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, category1s, category2s, similarities = search_similar_vectors(
        image_embedding_model_name,
        query_embeddings,
        category1,
        category2
    )
    return ids, similarities


if __name__ == "__main__":
    image_embedding_model_name = get_image_embedding_model_name()

    category1 = input("Enter category1 (or press Enter to skip): ").strip() or None
    category2 = input("Enter category2 (or press Enter to skip): ").strip() or None

    print_pgvector_info(image_embedding_model_name)
    image_embedding_model, params = load_image_embedding_model(image_embedding_model_name)
    all_ids, all_similarities = find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, params)
    save_retrieved_images_by_ids(image_embedding_model_name, all_ids, all_similarities, category1, category2)
