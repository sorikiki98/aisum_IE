import sys
from image_embedding_model import *
from vector_database import *
from PIL import Image
import shutil
from pgvector_database import *

def find_similar_product_ids(image_embedding_model,
                             image_embedding_model_name,
                             faiss_index_with_ids,
                             model_params=None):
    '''
    query_images = []
    query_images.append(article_image)
    '''
    query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, similarities = search_faiss_index(faiss_index_with_ids, query_embeddings, 3, 1024)

    return ids, similarities


def save_retrieved_images_by_ids(image_embedding_model_name, ids, similarities):
    for i, ids_per_one_query in enumerate(ids):
        retrieved_image_folder = f"./outputs/{image_embedding_model_name}/9"
        if os.path.exists(retrieved_image_folder):
            shutil.rmtree(retrieved_image_folder)
        os.makedirs(retrieved_image_folder, exist_ok=True)
        for j, image_id in enumerate(ids_per_one_query):
            file_path = os.path.join('./data/eseltree/images', str(image_id) + '.jpg')
            if os.path.exists(file_path):
                image = Image.open(file_path)
                save_path = os.path.join(retrieved_image_folder, f"top_{j + 1}_{similarities[i][j]}.jpg")
                image.save(save_path)


def find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, model_params=None):
    query_embeddings = embed_images(image_embedding_model, image_embedding_model_name, model_params)
    ids, category1s, category2s, similarities = search_similar_vectors(image_embedding_model_name, query_embeddings, category1, category2)
    
    return ids, similarities


if __name__ == "__main__":
    image_embedding_model_name = get_image_embedding_model_name()
    """
    faiss_index_with_ids = load_faiss_index(image_embedding_model_name)
    if faiss_index_with_ids is None:
        print("Faiss index file does not exist")
        sys.exit()
    print_faiss_index_info(faiss_index_with_ids)
    image_embedding_model, params = load_image_embedding_model(image_embedding_model_name)
    ids, similarities = find_similar_product_ids(image_embedding_model,
                                                 image_embedding_model_name,
                                                 faiss_index_with_ids,
                                                 params)
    """                                             
    category1 = input("Enter category1 (or press Enter to skip): ").strip() or None
    category2 = input("Enter category2 (or press Enter to skip): ").strip() or None
    print_pgvector_info(image_embedding_model_name)
    image_embedding_model, params = load_image_embedding_model(image_embedding_model_name)
    ids, similarities = find_similar_product(image_embedding_model, image_embedding_model_name, category1, category2, params)
    save_retrieved_images_by_ids(image_embedding_model_name, ids, similarities)
