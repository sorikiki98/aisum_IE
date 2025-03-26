import jax
import jax.numpy as jnp

from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np
import pickle
import timm

from flax import serialization
from models.resnet import ResNet152
from models.magiclens import MagicLens
from data.eseltree.dataset import EselTreeDatasetForMagicLens, EselTreeDatasetDefault
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from transformers import ViTModel, ViTImageProcessor


def get_num_dimensions_of_image_embedding_model(image_embedding_model_name):
    if image_embedding_model_name == "ViT":
        return 768
    elif image_embedding_model_name == "resnet152":
        return 2048
    elif image_embedding_model_name == "efnet":
        return 1000
    elif image_embedding_model_name == "magiclens_base":
        return 512
    elif image_embedding_model_name == "magiclens_large":
        return 1024
    elif image_embedding_model_name == "convnextv2_small":
        return 768
    elif image_embedding_model_name == "convnextv2_base":
        return 1024
    elif image_embedding_model_name == "convnextv2_large":
        return 1536
    else:
        raise ValueError("Invalid embedding model name")


def get_image_embedding_model_name():
    image_embedding_model_name = input("Enter embedding model name (ViT, resnet152, efnet, magiclens_base, "
                                       "magiclens_large, convnextv2_small, convnextv2_base, convnextv2_large): ")
    print(image_embedding_model_name)
    if image_embedding_model_name not in ["ViT", "efnet", "resnet152", "magiclens_base", "magiclens_large",
                                          "convnextv2_small",
                                          "convnextv2_base", "convnextv2_large"]:
        raise ValueError("Invalid embedding model name")
    return image_embedding_model_name


def get_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    return device


def load_image_embedding_model(image_embedding_model_name):
    if image_embedding_model_name == "ViT":
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "efnet":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "resnet152":
        model = ResNet152()
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "magiclens_base" or image_embedding_model_name == "magiclens_large":
        model_size = image_embedding_model_name.split("_")[1]
        model = MagicLens(model_size)
        rng = jax.random.PRNGKey(0)
        dummpy_input = {
            "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
            "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
        }
        params = model.init(rng, dummpy_input)
        with open(f"./models/magic_lens_clip_{model_size}.pkl", "rb") as f:
            model_bytes = pickle.load(f)
        params = serialization.from_bytes(params, model_bytes)
        params = jax.device_put(params)
        return model, params
    elif image_embedding_model_name == "convnextv2_small":
        model = timm.create_model('convnextv2_small', pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "convnextv2_base":
        model = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "convnextv2_large":
        model = timm.create_model('convnextv2_large.fcmae_ft_in1k', pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        device = get_device()
        model.to(device)
        model.eval()
        return model, None


def embed_images(image_embedding_model, image_embedding_model_name, model_params=None):
    tokenizer = clip_tokenizer.build_tokenizer()
    if image_embedding_model_name == "magiclens_base" or image_embedding_model_name == "magiclens_large":
        dataset = EselTreeDatasetForMagicLens(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)

        qimages = jnp.concatenate([i.qimage for i in query_examples], axis=0)
        qtokens = jnp.concatenate([i.qtokens for i in query_examples], axis=0)
        qembeds = image_embedding_model.apply(model_params, {"ids": qtokens, "image": qimages})[
            "multimodal_embed_norm"
        ]
        image_embeddings_ndarray = np.array(qembeds)
    elif image_embedding_model_name == "resnet152" or image_embedding_model_name == "efnet":
        tokenizer = clip_tokenizer.build_tokenizer()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)

        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "ViT":
        tokenizer = clip_tokenizer.build_tokenizer()
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [i.qimage['pixel_values'].to(get_device()) for i in query_examples]
        qimages = torch.cat(qimages, dim=0)

        with torch.no_grad():
            outputs = image_embedding_model(qimages)
        image_embeddings = outputs.last_hidden_state[:, 0, :]
        image_embeddings_ndarray = image_embeddings.cpu().numpy()
    elif image_embedding_model_name in ["convnextv2_large", "convnextv2_small", "convnextv2_base"]:
        tokenizer = clip_tokenizer.build_tokenizer()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(get_device())  # 1x1 크기로 변환
        qembeds = adaptive_pool(qembeds)
        qembeds = qembeds.reshape(qembeds.size(0), -1)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    else:
        raise ValueError(f"Invalid embedding model name: {image_embedding_model_name}")

    return image_embeddings_ndarray


def get_intermediate_embeddings(model, images, target_layer):
    feature_maps = get_intermediate_outputs(model, images, target_layer)
    image_embeddings = feature_maps_to_embeddings(feature_maps)
    return image_embeddings


def get_intermediate_outputs(model, images, target_layer):
    intermediate_outputs = None

    def hook(module, inputs, outputs):
        nonlocal intermediate_outputs
        intermediate_outputs = outputs

    layer = dict(model.features.named_children())[target_layer]
    hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        model(images)

    hook_handle.remove()
    return intermediate_outputs


def feature_maps_to_embeddings(feature_maps):
    device = get_device()
    global_average_pooling = nn.AdaptiveAvgPool2d((2, 2)).to(device)
    pooled_features = global_average_pooling(feature_maps)
    embedding_vectors = pooled_features.view(pooled_features.size(0), -1)
    return embedding_vectors
