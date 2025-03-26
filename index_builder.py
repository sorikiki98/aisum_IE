from transformers import ViTImageProcessor
from image_embedding_model import *
from data.eseltree.dataset import EselTreeDatasetForMagicLens, EselTreeDatasetDefault
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from vector_database import load_or_create_faiss_index, insert_image_embeddings_into_faiss_index, \
    save_faiss_index_to_disk, print_faiss_index_info
from pgvector_database import *
from pathlib import Path

def extract_last_two_categories(path_str):
    parts = Path(path_str).parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    elif len(parts) == 1:
        return None, parts[-1]  # category1 없음
    else:
        return None, None

if __name__ == "__main__":
    image_embedding_model_name = get_image_embedding_model_name()
    faiss_index_with_ids = load_or_create_faiss_index(image_embedding_model_name)

    params, dataset = None, None
    tokenizer = clip_tokenizer.build_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1024  # todo

    if image_embedding_model_name.startswith("magiclens"):
        image_embedding_model, params = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetForMagicLens(dataset_name="eseltree", tokenizer=tokenizer)
        category1, category2 = extract_last_two_categories(dataset.index_image_folder)
    elif image_embedding_model_name.startswith("convnextv2"):
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        category1, category2 = extract_last_two_categories(dataset.index_image_folder)
    elif image_embedding_model_name == 'ViT':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        category1, category2 = extract_last_two_categories(dataset.index_image_folder)
    else:
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        category1, category2 = extract_last_two_categories(dataset.index_image_folder)

    len_index_examples = len(dataset.index_image_ids)
    total_batches = len_index_examples // batch_size + (1 if len_index_examples % batch_size > 0 else 0)

    all_embeddings = []

    for batch_idx in range(total_batches):
        batch_files = dataset.index_image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = dataset.index_image_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        batch_examples = dataset.prepare_index_examples(batch_ids)

        if image_embedding_model_name.startswith("magiclens"):
            iimages = jnp.concatenate([i.iimage for i in batch_examples], axis=0)
            itokens = jnp.concatenate([i.itokens for i in batch_examples], axis=0)
            iembeds = image_embedding_model.apply(params, {"ids": itokens, "image": iimages})[
                "multimodal_embed_norm"
            ]
            batch_embeddings_ndarray = jax.device_get(iembeds)
            batch_embedding_ndarray = np.ascontiguousarray(batch_embeddings_ndarray, dtype=np.float32)
            batch_ids = np.ascontiguousarray(np.array(batch_ids), dtype=np.int64)
        elif image_embedding_model_name.startswith("convnextv2"):
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            batch_embeddings = adaptive_pool(batch_embeddings)
            batch_embeddings = batch_embeddings.reshape(batch_embeddings.size(0), -1)
            print(batch_embeddings.shape)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
            batch_ids = np.ascontiguousarray(np.array(batch_ids), dtype=np.int64)
        elif image_embedding_model_name == 'ViT':
            iimages = [i.iimage['pixel_values'].to(device) for i in batch_examples]
            iimages = torch.cat(iimages, dim=0)
            with torch.no_grad():
                outputs = image_embedding_model(iimages)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
            batch_ids = np.ascontiguousarray(np.array(batch_ids), dtype=np.int64)
        else:
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
            batch_ids = np.ascontiguousarray(np.array(batch_ids), dtype=np.int64)
        #insert_image_embeddings_into_faiss_index(faiss_index_with_ids, batch_embeddings_ndarray, batch_ids)
        #save_faiss_index_to_disk(faiss_index_with_ids, image_embedding_model_name)
        #print_faiss_index_info(faiss_index_with_ids)

        insert_image_embeddings_into_postgres(image_embedding_model_name, batch_ids, batch_embeddings_ndarray, category1, category2, batch_size)