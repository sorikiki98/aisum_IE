import jax
import jax.numpy as jnp

from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import pickle
import timm

from flax import serialization
from models.resnet import ResNet152
from models.resnext import ResNext101
from models.magiclens import MagicLens
from dataset import EselTreeDatasetForMagicLens, EselTreeDatasetDefault
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from transformers import ViTModel, ViTImageProcessor


def get_num_dimensions_of_image_embedding_model(image_embedding_model_name):
    if image_embedding_model_name == "ViT":
        return 768
    elif image_embedding_model_name == "resnet152" or image_embedding_model_name == "resnext101":
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
                                       "magiclens_large, convnextv2_small, convnextv2_base, convnextv2_large, "
                                       "resnext101): ")
    print(image_embedding_model_name)
    if image_embedding_model_name not in ["ViT", "efnet", "resnet152", "magiclens_base", "magiclens_large",
                                          "convnextv2_small", "convnextv2_base", "convnextv2_large",
                                          "resnext101"]:
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
    elif image_embedding_model_name == "resnext101":
        model = ResNext101()
        device = get_device()
        model.to(device)
        model.eval()
        return model, None


def embed_images(image_embedding_model, image_embedding_model_name, model_params=None, query_image_bytes=None):
    tokenizer = clip_tokenizer.build_tokenizer()
    device = get_device()

    if query_image_bytes is not None:
        # 🔹 단일 이미지 바이트 직접 처리하는 흐름 (배치 1)
        from PIL import Image
        import io
        import torchvision.transforms as T

        image = Image.open(io.BytesIO(query_image_bytes[0])).convert("RGB")  # 리스트로 감싼 한 장
        print(f"✅ 단일 이미지 처리 - 모델: {image_embedding_model_name}")

        if image_embedding_model_name in ["resnet152", "resnext101", "efnet"]:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = image_embedding_model(image_tensor)
            return embedding.cpu().numpy()

        elif image_embedding_model_name.startswith("convnextv2"):
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = image_embedding_model(image_tensor)

            adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
            embedding = adaptive_pool(embedding)
            embedding = embedding.reshape(embedding.size(0), -1)

            return embedding.cpu().numpy()

        elif image_embedding_model_name == "ViT":
            from transformers import ViTImageProcessor
            preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            inputs = preprocess(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = image_embedding_model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
            return embedding.cpu().numpy()

        else:
            raise ValueError(f"단일 이미지 처리 미지원 모델: {image_embedding_model_name}")
    else:
        # ✅ 기존처럼 Dataset 기반 로딩
        if image_embedding_model_name == "magiclens_base" or image_embedding_model_name == "magiclens_large":
            dataset = EselTreeDatasetForMagicLens(dataset_name="eseltree", tokenizer=tokenizer)
            query_ids = dataset.query_image_ids
            query_examples = dataset.prepare_query_examples(query_ids)

            qimages = jnp.concatenate([i.qimage for i in query_examples], axis=0)
            qtokens = jnp.concatenate([i.qtokens for i in query_examples], axis=0)
            qembeds = image_embedding_model.apply(model_params, {"ids": qtokens, "image": qimages})[
                "multimodal_embed_norm"
            ]
            return np.array(qembeds)

        else:
            dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
            query_ids = dataset.query_image_ids
            query_examples = dataset.prepare_query_examples(query_ids)

            if image_embedding_model_name == "ViT":
                qimages = [i.qimage['pixel_values'].to(device) for i in query_examples]
                qimages = torch.cat(qimages, dim=0)
                with torch.no_grad():
                    outputs = image_embedding_model(qimages)
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()

            else:
                qimages = [q.qimage for q in query_examples]
                qimages = torch.stack(qimages).to(device)
                with torch.no_grad():
                    embedding = image_embedding_model(qimages)

                if image_embedding_model_name.startswith("convnextv2"):
                    adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
                    embedding = adaptive_pool(embedding)
                    embedding = embedding.reshape(embedding.size(0), -1)

                return embedding.cpu().numpy()



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
