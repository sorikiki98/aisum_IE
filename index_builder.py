import json
import importlib
import sys
from dataset import IndexDataset
from pgvector_database import PGVectorDB
from model_utils import ImageEmbeddingModel


def load_image_embedding_model_from_path(model_name: str, cfg: dict):
    model_cfg = cfg["model"][model_name]
    module = importlib.import_module(model_cfg["model_dir"])
    class_name = model_cfg["model_name"]
    cls = getattr(module, class_name)

    if not issubclass(cls, ImageEmbeddingModel):
        raise TypeError(f"{class_name} does not inherit from ImageEmbeddingModel")

    return cls(model_name, cfg)


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])

    image_embedding_model_name = input("Enter embedding model name: ")

    if image_embedding_model_name not in config["model"]:
        raise ValueError("Invalid embedding model name.")

    dataset = IndexDataset("eseltree", config)
    database = PGVectorDB(image_embedding_model_name, config)

    model = load_image_embedding_model_from_path(image_embedding_model_name, config)
    batch_size = config["model"][image_embedding_model_name]["batch_size"]

    len_index_images = len(dataset.index_image_ids)
    total_batches = len_index_images // batch_size + (1 if len_index_images % batch_size > 0 else 0)
    all_embeddings = []

    for batch_idx in range(total_batches):
        batch_images, batch_ids, batch_cat1s, batch_cat2s = dataset.prepare_index_images(batch_idx, batch_size)
        batch_embeddings_ndarray = model(batch_images)
        database.insert_image_embeddings_into_postgres(batch_ids, batch_embeddings_ndarray,
                                                       batch_cat1s, batch_cat2s)
    '''
    if image_embedding_model_name.startswith("convnextv2"):
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer,
                                         image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'vit':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess,
                                         image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == "swin":
        image_embedding_model, preprocess = load_image_embedding_model("swin_base_patch4_window7_224")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'blip2':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xxl')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess,
                                         prompt="Question: Describe the product. Answer:"
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'openai_clip':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = image_embedding_model.preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'laion_clip':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = image_embedding_model.preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'dinov2':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=None
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'siglip_so400':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'siglip_large':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == 'siglip2':
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        preprocess = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == "imagebind":
        image_embedding_model, preprocess = load_image_embedding_model("imagebind")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == "mobilenetv3":
        image_embedding_model, preprocess = load_image_embedding_model("mobilenetv3")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess
                                         , image_embedding_model_name=image_embedding_model_name)
    elif image_embedding_model_name == "fashionclip":
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer
                                         , image_embedding_model_name=image_embedding_model_name)
    else:
        image_embedding_model, _ = load_image_embedding_model(image_embedding_model_name)
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer
                                         , image_embedding_model_name=image_embedding_model_name)

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
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1,
                                                                            keepdim=True)  # (optional) cosine similarity정규화
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == "swin":
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name == "imagebind":
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            inputs = {ModalityType.VISION: iimages}
            with torch.no_grad():
                batch_embeddings = image_embedding_model(inputs)
                batch_embeddings = batch_embeddings[ModalityType.VISION]
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model == "mobilenetv3":
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
        elif image_embedding_model_name == "dinov2":
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device).half()
            image_embedding_model = image_embedding_model.half()
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        elif image_embedding_model_name.startswith("siglip"):
            batch_images = [i.iimage for i in batch_examples]
            inputs = preprocess(images=batch_images, return_tensors="pt", padding=True)
            text_inputs = tokenizer([""] * len(batch_images), return_tensors="pt", padding=True)
            inputs.update(text_inputs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = image_embedding_model(**inputs)
            batch_embeddings_ndarray = outputs.image_embeds.cpu().numpy()
        elif image_embedding_model_name == "fashionclip":
            iimages = [i.iimage for i in batch_examples]
            batch_embeddings_ndarray = image_embedding_model.encode_images(iimages)
            batch_embeddings_ndarray = np.array(batch_embeddings_ndarray, dtype=np.float32)
        else:
            iimages = [i.iimage for i in batch_examples]
            iimages = torch.stack(iimages).to(device)
            with torch.no_grad():
                batch_embeddings = image_embedding_model(iimages)
            batch_embeddings_ndarray = batch_embeddings.cpu().numpy()
        '''
