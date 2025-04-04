import os
import shutil
from PIL import Image
from pathlib import Path
from image_embedding_model import *
from pgvector_database import *


def save_retrieved_images_by_ids(image_embedding_model_name, all_batch_ids, all_batch_similarities, category1,
                                 category2):
    project_root = Path(__file__).parent
    data_root = project_root / "data" / "eseltree" / "images"

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

                if original_path != public_image_path:
                    public_image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(original_path), str(public_image_path))
            else:
                print(f"❌ 원본 이미지 없음: {original_path}")


def find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, model_params=None):
    query_ids, query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, category1s, category2s, similarities = search_similar_vectors(
        image_embedding_model_name=image_embedding_model_name,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        category1=category1,
        category2=category2
    )
    return ids, similarities


def main(model_name=None, category1=None, category2=None):
    # For web interface, model_name is required
    if model_name is None:
        raise ValueError("model_name is required")

    print_pgvector_info(model_name)
    image_embedding_model, params = load_image_embedding_model(model_name)
    all_ids, all_similarities = find_similar_product(image_embedding_model, model_name, category1, category2, params)
    save_retrieved_images_by_ids(model_name, all_ids, all_similarities, category1, category2)

    return {
        'result_ids': all_ids,
        'result_distances': all_similarities
    }
