from image_embedding_model import *
from dataset import EselTreeDatasetForMagicLens, EselTreeDatasetDefault
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from pgvector_database import *
from pathlib import Path
from transformers import Blip2Processor

import sys

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
    # faiss_index_with_ids = load_or_create_faiss_index(image_embedding_model_name)

    params, dataset = None, None
    tokenizer = clip_tokenizer.build_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    if image_embedding_model_name.startswith("magiclens"):
        image_embedding_model, params = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetForMagicLens(dataset_name="eseltree", tokenizer=tokenizer)
    elif image_embedding_model_name.startswith("convnextv2"):
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
    elif image_embedding_model_name == 'vit':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
    elif image_embedding_model_name == "unicom_all":
        image_embedding_model, preprocess = load_image_embedding_model(image_embedding_model_name)  # 모델명 선택 가능
        image_embedding_model = image_embedding_model.to(device)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
    elif image_embedding_model_name == "swin":
        image_embedding_model, preprocess = load_image_embedding_model("swin_base_patch4_window7_224")
    elif image_embedding_model_name == 'blip2':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xxl')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess,
                                         prompt="Question: Describe the product. Answer:")
    elif image_embedding_model_name == 'openai_clip':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = image_embedding_model.preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
    elif image_embedding_model_name == 'laion_clip':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = image_embedding_model.preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
    else:
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)

    len_index_examples = len(dataset.index_image_ids)
    total_batches = len_index_examples // batch_size + (1 if len_index_examples % batch_size > 0 else 0)
    all_embeddings = []

    for batch_idx in range(total_batches):
        batch_files = dataset.index_image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids = dataset.index_image_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_ids_with_cats = dataset.index_image_ids_with_cats[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        batch_examples = dataset.prepare_index_examples(batch_ids_with_cats)
        batch_ids = np.ascontiguousarray(np.array(batch_ids), dtype=np.int64)
        batch_cat1s = [i.category1_code for i in batch_examples]
        batch_cat2s = [i.category2_code for i in batch_examples]

        if image_embedding_model_name.startswith("magiclens"):
            iimages = jnp.concatenate([i.iimage for i in batch_examples], axis=0)
            itokens = jnp.concatenate([i.itokens for i in batch_examples], axis=0)
            iembeds = image_embedding_model.apply(params, {"ids": itokens, "image": iimages})[
                "multimodal_embed_norm"
            ]
            batch_embeddings_ndarray = jax.device_get(iembeds)
            batch_embeddings_ndarray = np.ascontiguousarray(batch_embeddings_ndarray, dtype=np.float32)
        elif image_embedding_model_name.startswith("convnextv2"):
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            batch_embeddings = adaptive_pool(batch_embeddings)
            batch_embeddings = batch_embeddings.reshape(batch_embeddings.size(0), -1)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == 'vit':
            iimages = [i.iimage['pixel_values'].to(device) for i in batch_examples]
            iimages = torch.cat(iimages, dim=0)
            with torch.no_grad():
                outputs = image_embedding_model(iimages)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == 'efnet':
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == "unicom_all":
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)  # (optional) cosine similarity정규화
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == "swin":
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == 'blip2':
            iimages = [i.iimage['pixel_values'].to(device) for i in batch_examples]
            iimages = torch.cat(iimages, dim=0)
            with torch.no_grad():
                outputs = image_embedding_model.get_qformer_features(iimages).last_hidden_state
                pooled_outputs = torch.mean(outputs, dim=1)
            batch_embeddings_ndarray = pooled_outputs.cpu().numpy()
        elif image_embedding_model_name == 'openai_clip':
            iimages = [i.iimage for i in batch_examples]
            iimage_tensors = torch.stack(iimages).to(device)
            batch_embeddings_ndarray = image_embedding_model.embed_images(iimage_tensors).cpu().numpy()
        elif image_embedding_model_name == 'laion_clip':
            iimages = [i.iimage for i in batch_examples]
            iimage_tensors = torch.stack(iimages).to(device)
            batch_embeddings_ndarray = image_embedding_model.embed_images(iimage_tensors).cpu().numpy()
        else:
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        # insert_image_embeddings_into_faiss_index(faiss_index_with_ids, batch_embeddings_ndarray, batch_ids)
        # save_faiss_index_to_disk(faiss_index_with_ids, image_embedding_model_name)
        # print_faiss_index_info(faiss_index_with_ids)
        insert_image_embeddings_into_postgres(image_embedding_model_name, batch_ids, batch_embeddings_ndarray,
                                              batch_cat1s, batch_cat2s)