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


def save_retrieved_images_by_ids(image_embedding_model_name, all_batch_ids, all_batch_similarities,
                                 all_batch_cat1s, all_batch_cat2s):
    saved_paths = []
    
    for batch_index, (batch_ids, cat1_list, cat2_list, sim_list) in enumerate(
        zip(all_batch_ids, all_batch_cat1s, all_batch_cat2s, all_batch_similarities)):
        
        batch_paths = [] 
        retrieved_image_folder = f"../outputs/{image_embedding_model_name}/{batch_index}"

        if os.path.exists(retrieved_image_folder):
            shutil.rmtree(retrieved_image_folder)
        os.makedirs(retrieved_image_folder, exist_ok=True)

        for idx, (img_id, cat1, cat2, similarity) in enumerate(zip(batch_ids, cat1_list, cat2_list, sim_list)):
            file_path = os.path.join("../data/eseltree/images",cat1,cat2,f"{img_id}.jpg")
            save_name = f"top_{idx + 1}_{similarity}.jpg"
            save_path = os.path.join(retrieved_image_folder, save_name)

            if os.path.exists(file_path):
                image = Image.open(file_path)
                image.save(save_path)
                relative_path = f"outputs/{image_embedding_model_name}/{batch_index}/{save_name}"
                batch_paths.append(relative_path)
            else:
                print(f"File does not exist: {file_path}")
                
        saved_paths.append(batch_paths)
    
    return saved_paths


def find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, model_params=None):
    query_ids, query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, category1s, category2s, similarities = search_similar_vectors(
        image_embedding_model_name=image_embedding_model_name,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        category1=category1,
        category2=category2
    )
    return ids, similarities, category1s, category2s


def main(model_name=None, category1=None, category2=None):
    if model_name is None:
        raise ValueError("model_name is required")

    category1 = category1 if category1 and category1.strip() else None
    category2 = category2 if category2 and category2.strip() else None

    print_pgvector_info(model_name)
    image_embedding_model, params = load_image_embedding_model(model_name)
    all_ids, all_similarities, all_cat1s, all_cat2s = find_similar_product(
        image_embedding_model, 
        model_name, 
        category1, 
        category2,  
        params
    )
    result_paths = save_retrieved_images_by_ids(model_name, all_ids, all_similarities, all_cat1s, all_cat2s)
    
    return {
        'result_ids': all_ids,
        'result_distances': all_similarities,
        'result_paths': result_paths
    }
