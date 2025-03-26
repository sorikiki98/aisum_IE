import os
import faiss
import pickle
import numpy as np
from image_embedding_model import get_num_dimensions_of_image_embedding_model


def create_faiss_index(num_dimensions, M=32, efConstruction=200):
    faiss_index = faiss.IndexHNSWFlat(num_dimensions, M, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efConstruction = efConstruction
    return faiss.IndexIDMap(faiss_index)


def get_faiss_index_file_name(image_embedding_model_name):
    return f"./faiss_index_{image_embedding_model_name}"


def load_faiss_index(image_embedding_model_name):
    index_file_name = get_faiss_index_file_name(image_embedding_model_name)
    if os.path.exists(index_file_name):
        return faiss.read_index(index_file_name)
    else:
        return None


def load_or_create_faiss_index(image_embedding_model_name):
    index = load_faiss_index(image_embedding_model_name)
    if index is None:
        num_dimensions = get_num_dimensions_of_image_embedding_model(image_embedding_model_name)
        index = create_faiss_index(num_dimensions)
    print_faiss_index_info(index)
    return index


def save_faiss_index_to_disk(index, model_name):
    faiss.write_index(index, get_faiss_index_file_name(model_name))


def insert_image_embeddings_into_faiss_index(index, embeddings, ids):
    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings)
    ids = np.array(ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids)


def search_faiss_index(index, query, k, efSearch):
    faiss.normalize_L2(query)
    faiss_index = faiss.downcast_index(index.index)
    faiss_index.hnsw.efSearch = efSearch
    similarities, ids = index.search(query, k)
    return ids, similarities


def print_faiss_index_info(index):
    d = index.index.d
    print("--- Faiss index info ---")
    print(f"dimensions: {d}")
    print(f"total vectors: {index.ntotal}")
    print("------------------------")


def get_max_id_from_faiss_index(index):
    id_array = faiss.vector_to_array(index.id_map)
    return 0 if id_array.size == 0 else max(id_array)


def get_min_id_from_faiss_index(index):
    id_array = faiss.vector_to_array(index.id_map)
    return 0 if id_array.size == 0 else min(id_array)


def get_unique_id_count_from_faiss_index(index):
    id_array = faiss.vector_to_array(index.id_map)
    return len(np.unique(id_array))


def get_id_count_from_faiss_index(index):
    return index.ntotal


# ✅ 추가된 함수: FastAPI용 search wrapper
def search_similar_images(query_vector: np.ndarray, model_name: str, top_k=10):
    faiss_index = load_faiss_index(model_name)
    if faiss_index is None:
        raise FileNotFoundError("FAISS index not found")

    query_vector = np.expand_dims(query_vector, axis=0).astype("float32")
    ids, _ = search_faiss_index(faiss_index, query_vector, k=top_k, efSearch=256)

    with open(f"./image_paths_{model_name}.pkl", "rb") as f:
        path_map = pickle.load(f)

    results = []
    for id in ids[0]:
        path = path_map.get(id, "UNKNOWN").replace("\\", "/")
        if path.startswith("/images"):
            results.append(path)
        else:
            results.append(f"/images/{path}")
    return results
