import os
import faiss
import numpy as np
from image_embedding_model import get_num_dimensions_of_image_embedding_model


# M: the number of neighbors used in the HNSW graph (32~64)
# efConstruction: the depth of exploration at add time (200~500)
def create_faiss_index(num_dimensions, M=32, efConstruction=200):
    faiss_index = faiss.IndexHNSWFlat(num_dimensions, M, faiss.METRIC_INNER_PRODUCT)
    # [Important Note] We use "cosine similarity" (the more similar, the larger)
    # We must normalize the vectors prior to adding them to the index (with faiss.normalize_L2)
    # We must normalize the vectors to searching them
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    faiss_index.hnsw.efConstruction = efConstruction
    faiss_index_with_ids = faiss.IndexIDMap(faiss_index)  # id -> piclick.product_list2.id

    return faiss_index_with_ids


def load_faiss_index(image_embedding_model_name):
    index_file_name = get_faiss_index_file_name(image_embedding_model_name)
    if os.path.exists(index_file_name):
        return faiss.read_index(index_file_name)
    else:
        return None


def get_faiss_index_file_name(image_embedding_model_name):
    if image_embedding_model_name == "ViT":
        return "./index/faiss_index_ViT"
    elif image_embedding_model_name == "resnet152":
        return "./index/faiss_index_resnet152"
    elif image_embedding_model_name == "efnet":
        return "./index/faiss_index_efnet"
    elif image_embedding_model_name == "magiclens_base":
        return "./index/faiss_index_magiclens_base"
    elif image_embedding_model_name == "magiclens_large":
        return "./index/faiss_index_magiclens_large"
    elif image_embedding_model_name == "convnextv2_base":
        return "./index/faiss_index_convnextv2_base"
    elif image_embedding_model_name == "convnextv2_large":
        return "./index/faiss_index_convnextv2_large"
    else:
        raise ValueError("Invalid embedding model name")


def load_or_create_faiss_index(image_embedding_model_name):
    faiss_index_with_ids = load_faiss_index(image_embedding_model_name)
    if faiss_index_with_ids is None:
        num_dimensions = get_num_dimensions_of_image_embedding_model(image_embedding_model_name)
        faiss_index_with_ids = create_faiss_index(num_dimensions)
    print_faiss_index_info(faiss_index_with_ids)
    return faiss_index_with_ids


def save_faiss_index_to_disk(faiss_index_with_ids, image_embedding_model_name):
    index_file_name = get_faiss_index_file_name(image_embedding_model_name)
    faiss.write_index(faiss_index_with_ids, index_file_name)


def get_max_id_from_faiss_index(faiss_index_with_ids):
    id_array = faiss.vector_to_array(faiss_index_with_ids.id_map)
    return 0 if id_array.size == 0 else max(id_array)


def get_min_id_from_faiss_index(faiss_index_with_ids):
    id_array = faiss.vector_to_array(faiss_index_with_ids.id_map)
    return 0 if id_array.size == 0 else min(id_array)


def get_unique_id_count_from_faiss_index(faiss_index_with_ids):
    id_array = faiss.vector_to_array(faiss_index_with_ids.id_map)
    return len(np.unique(id_array))


def get_id_count_from_faiss_index(faiss_index_with_ids):
    return faiss_index_with_ids.ntotal


def print_faiss_index_info(faiss_index_with_ids):
    dimensions = faiss_index_with_ids.index.d
    id_count = get_id_count_from_faiss_index(faiss_index_with_ids)
    unique_id_count = get_unique_id_count_from_faiss_index(faiss_index_with_ids)
    max_id = get_max_id_from_faiss_index(faiss_index_with_ids)
    min_id = get_min_id_from_faiss_index(faiss_index_with_ids)

    print("--- Faiss index information ---")
    print(f"dimensions: {dimensions}")
    print(f"id count: {id_count}")
    print(f"unique id count: {unique_id_count}")
    print(f"min id: {min_id}")
    print(f"max id: {max_id}")
    print("-------------------------------")


# image_embeddings: numpy.ndarray (shape: (n, dimensions))
# ids: list[int]
def insert_image_embeddings_into_faiss_index(faiss_index_with_ids, image_embeddings, ids):
    image_embedding_ndarray = np.ascontiguousarray(image_embeddings)
    faiss.normalize_L2(image_embedding_ndarray)  # for cosine similarity
    id_ndarray = np.array(ids)  # list[int] -> numpy.ndarray
    faiss_index_with_ids.add_with_ids(image_embedding_ndarray, id_ndarray)


# query_embeddings: numpy.ndarray (shape: (n, dimensions))
# k: the number of the nearest neighbors
# efSearch: the depth of exploration of the search (300~500)
def search_faiss_index(faiss_index_with_ids, query_embeddings, k, efSearch):
    # query_embeddings = np.array(query_embeddings)  # query_embeddings: numpy.ndarray (shape: (n, dimensions))
    faiss.normalize_L2(query_embeddings)  # for cosine similarity
    faiss_index = faiss.downcast_index(faiss_index_with_ids.index)
    faiss_index.hnsw.efSearch = efSearch

    similarities, ids = faiss_index_with_ids.search(query_embeddings, 10)
    return ids, similarities
