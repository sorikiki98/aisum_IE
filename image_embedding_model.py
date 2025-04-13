import pickle
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import timm
from timm import create_model
from timm.data import create_transform

from flax import serialization
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

from transformers import (
    ViTModel, 
    ViTImageProcessor, 
    Blip2Processor,
    Blip2Model,
    AutoProcessor, 
    AutoModel, 
    AutoTokenizer
)

import unicom
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from models.resnet import ResNet152
from models.resnext import ResNext101
from models.magiclens import MagicLens
from models.openai_clip import OpenAICLIP
from models.laion_clip import LaionCLIP
from models.fashion_clip import FashionCLIP
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from dataset import EselTreeDatasetForMagicLens, EselTreeDatasetDefault

warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_num_dimensions_of_image_embedding_model(image_embedding_model_name):
    if image_embedding_model_name == "vit":
        return 768
    elif image_embedding_model_name == "resnet152" or image_embedding_model_name == "resnext101":
        return 2048
    elif image_embedding_model_name == "efnet":
        return 1000
    elif image_embedding_model_name == "magiclens_base":
        return 512
    elif image_embedding_model_name == "magiclens_large":
        return 768
    elif image_embedding_model_name == "convnextv2_small":
        return 768
    elif image_embedding_model_name == "convnextv2_base":
        return 1024
    elif image_embedding_model_name == "convnextv2_large":
        return 1536
    elif image_embedding_model_name == "unicom_all":
        return 512
    elif image_embedding_model_name == "swin":
        return 1024
    elif image_embedding_model_name == "blip2":
        return 768
    elif image_embedding_model_name == "openai_clip":
        return 768
    elif image_embedding_model_name == "laion_clip":
        return 1024
    elif image_embedding_model_name == "dinov2":
        return 1536
    elif image_embedding_model_name == "siglip_so400m":
        return 1152
    elif image_embedding_model_name == "siglip_large":
        return 1024
    elif image_embedding_model_name == "siglip2":
        return 1152
    elif image_embedding_model_name == "imagebind":
        return 1024
    elif image_embedding_model_name == "mobilenetv3":
        return 1000
    elif image_embedding_model_name == "fashionclip":
        return 512
    elif image_embedding_model_name == "sam2":
        return 256
    else:
        raise ValueError("Invalid embedding model name")


def get_image_embedding_model_name():
    image_embedding_model_name = input("Enter embedding model name (vit, resnet152, efnet, magiclens_base, "
                                       "magiclens_large, convnextv2_small, convnextv2_base, convnextv2_large, "
                                       "resnext101, unicom_all, swin, blip2, openai_clip, laion_clip, "
                                       "dinov2, siglip_so400m, siglip_large, siglip2, imagebind, mobilenetv3, fashionclip, sam2): ")
    if image_embedding_model_name not in ["vit", "efnet", "resnet152", "magiclens_base", "magiclens_large",
                                          "convnextv2_small", "convnextv2_base", "convnextv2_large",
                                          "resnext101", "unicom_all", "swin", "blip2", "openai_clip", "laion_clip",
                                          "dinov2", "siglip_so400m", "siglip_large", "siglip2", "imagebind",
                                          "mobilenetv3", "fashionclip", "sam2"]:
        raise ValueError("Invalid embedding model name")
    return image_embedding_model_name


def get_device():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    return device


def load_image_embedding_model(image_embedding_model_name):
    print(f"📌 모델 이름 들어옴: {image_embedding_model_name}")
    if image_embedding_model_name == "vit":
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
        with open(f"../models/magic_lens_clip_{model_size}.pkl", "rb") as f:
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
    elif image_embedding_model_name == "unicom_all":
        model, preprocess = unicom.load("ViT-B/32")
        device = get_device()
        model.to(device)
        model.eval()
        return model, preprocess
    elif image_embedding_model_name == "swin_base_patch4_window7_224" or image_embedding_model_name == "swin":
        model = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)  # classification head 제거
        device = get_device()
        model.to(device)
        model.eval()
        transform = create_transform(
            input_size=(3, 224, 224),
            is_training=False
        )
        return model, transform
    elif image_embedding_model_name == "blip2":
        model = Blip2Model.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16
        )
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=True, force_reload=False,
                               verbose=False)
        for block in model.blocks:
            block.attn.use_mem_efficient_attention = True
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "openai_clip":
        model = OpenAICLIP()
        return model, None
    elif image_embedding_model_name == "laion_clip":
        model = LaionCLIP()
        return model, None
    elif image_embedding_model_name == "siglip_so400m":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "siglip_large":
        model = AutoModel.from_pretrained("google/siglip-large-patch16-384")
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "siglip2":
        model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "imagebind":
        model = imagebind_model.imagebind_huge(pretrained=True)
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "mobilenetv3":
        model = mobilenet_v3_large(weights=None)
        model = model.to("cuda")

        state_dict = torch.load("/home/jiwoo/magiclens/aisum_IE/models/mobilenet_v3_large-5c1a4163.pth",
                                map_location="cuda")
        model.load_state_dict(state_dict)
        device = get_device()
        model.to(device)
        model.eval()
        return model, None
    elif image_embedding_model_name == "fashionclip":
        device = get_device()
        model = FashionCLIP("patrickjohncyh/fashion-clip", device=device)
        return model, None
    elif image_embedding_model_name == "sam2":
        model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
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
    elif image_embedding_model_name == "resnet152" or image_embedding_model_name == "resnext101" or image_embedding_model_name == "efnet":
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "vit":
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        #preprocess 버전 2
        qimages = [torch.from_numpy(i.qimage['pixel_values'][0]) for i in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        #preprocess 버전 1
        #qimages = [i.qimage['pixel_values'].to(get_device()) for i in query_examples]
        with torch.no_grad():
            outputs = image_embedding_model(qimages)
        image_embeddings = outputs.last_hidden_state[:, 0, :]
        image_embeddings_ndarray = image_embeddings.cpu().numpy()
    elif image_embedding_model_name in ["convnextv2_large", "convnextv2_small", "convnextv2_base"]:
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
    elif image_embedding_model_name == "unicom_all":
        tokenizer = clip_tokenizer.build_tokenizer()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=model_params)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
            qembeds = qembeds / qembeds.norm(dim=-1, keepdim=True)  # optional: 정규화
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "swin":
        tokenizer = clip_tokenizer.build_tokenizer()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=model_params)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "blip2":
        preprocess = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xxl')
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess,
                                         prompt="Question: Describe the product. Answer:")
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [i.qimage['pixel_values'].to(get_device()) for i in query_examples]
        qimages = torch.cat(qimages, dim=0)
        with torch.no_grad():
            outputs = image_embedding_model.get_qformer_features(qimages).last_hidden_state
            pooled_outputs = torch.mean(outputs, dim=1)
        image_embeddings_ndarray = pooled_outputs.cpu().numpy()
    elif image_embedding_model_name == "openai_clip":
        preprocess = OpenAICLIP().preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [i.qimage for i in query_examples]
        qimage_tensors = torch.stack(qimages).to(get_device())
        image_embeddings_ndarray = image_embedding_model.embed_images(qimage_tensors).cpu().numpy()
    elif image_embedding_model_name == "laion_clip":
        preprocess = LaionCLIP().preprocess
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [i.qimage for i in query_examples]
        qimage_tensors = torch.stack(qimages).to(get_device())
        image_embeddings_ndarray = image_embedding_model.embed_images(qimage_tensors).cpu().numpy()
    elif image_embedding_model_name == "dinov2":
        tokenizer = clip_tokenizer.build_tokenizer()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device()).half()  # float16으로 변환
        image_embedding_model = image_embedding_model.half()  # 모델도 float16으로 변환
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "siglip_so400m":
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        inputs = preprocess(images=qimages, return_tensors="pt", padding=True)
        text_inputs = tokenizer([""] * len(qimages), return_tensors="pt", padding=True)
        inputs.update(text_inputs)
        inputs = {k: v.to(get_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = image_embedding_model(**inputs)
        image_embeddings_ndarray = outputs.image_embeds.cpu().numpy()
    elif image_embedding_model_name == "siglip_large":
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-large-patch16-384")
        preprocess = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        inputs = preprocess(images=qimages, return_tensors="pt", padding=True)
        text_inputs = tokenizer([""] * len(qimages), return_tensors="pt", padding=True)
        inputs.update(text_inputs)
        inputs = {k: v.to(get_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = image_embedding_model(**inputs)
        image_embeddings_ndarray = outputs.image_embeds.cpu().numpy()
    elif image_embedding_model_name == "siglip2":
        tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384")
        preprocess = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        inputs = preprocess(images=qimages, return_tensors="pt", padding=True)
        text_inputs = tokenizer([""] * len(qimages), return_tensors="pt", padding=True)
        inputs.update(text_inputs)
        inputs = {k: v.to(get_device()) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = image_embedding_model(**inputs)
        image_embeddings_ndarray = outputs.image_embeds.cpu().numpy()
    elif image_embedding_model_name == "imagebind":
        def preprocess(path, **kwargs):
            return data.load_and_transform_vision_data([path], device=get_device())
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model({ModalityType.VISION: qimages})
            qembeds = qembeds[ModalityType.VISION]
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "mobilenetv3":
        preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qimages = torch.stack(qimages).to(get_device())
        with torch.no_grad():
            qembeds = image_embedding_model(qimages)
        image_embeddings_ndarray = qembeds.cpu().numpy()
    elif image_embedding_model_name == "fashionclip":
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        qembeds = image_embedding_model.encode_images(qimages)  
        image_embeddings_ndarray = np.array(qembeds, dtype=np.float32)
    elif image_embedding_model_name == "sam2":
        preprocess = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        dataset = EselTreeDatasetDefault(dataset_name="eseltree", tokenizer=tokenizer, preprocess=preprocess)
        query_ids = dataset.query_image_ids
        query_examples = dataset.prepare_query_examples(query_ids)
        qimages = [q.qimage for q in query_examples]
        embeddings = []
        with torch.no_grad():
            for image in qimages:
                image_embedding_model.set_image(image)
                features = image_embedding_model.get_image_embedding()
                features = features.mean(dim=(-2, -1))
                features = features.squeeze(0)
                features = F.normalize(features, p=2, dim=0)
                embedding = features.detach().cpu().numpy()
                embeddings.append(embedding)
        image_embeddings_ndarray = np.stack(embeddings)
        return query_ids, image_embeddings_ndarray
    else:
        raise ValueError(f"Invalid embedding model name: {image_embedding_model_name}")

    return query_ids, image_embeddings_ndarray


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