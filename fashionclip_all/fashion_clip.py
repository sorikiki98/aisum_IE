import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import List, Union
from transformers import CLIPModel, CLIPProcessor  

class FashionCLIP:
    def __init__(self, model_name="patrickjohncyh/fashion-clip", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_images(self, images: List[Union[Image.Image, torch.Tensor]], batch_size: int = 32) -> np.ndarray:
        """
        이미지 리스트를 받아 임베딩 반환 (torch.Tensor 또는 PIL.Image.Image 허용)
        HuggingFace Datasets 제거 버전
                """

        def preprocess(img):
            if isinstance(img, torch.Tensor):
                # 이미지: C x H x W → H x W x C
                img = img.permute(1, 2, 0).cpu().numpy()

                # 값 범위 [-1, 1] → [0, 255]
                img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

                # numpy → PIL 변환
                img = Image.fromarray(img)

            return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)


        preprocessed = [preprocess(img) for img in images]  # [C, H, W] 텐서 리스트
        image_tensor = torch.stack(preprocessed)  # [N, C, H, W]
        dataloader = torch.utils.data.DataLoader(image_tensor, batch_size=batch_size)

        image_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="🔄 Encoding Images"):
                batch = batch.to(self.device)
                embeds = self.model.get_image_features(pixel_values=batch)
                embeds = F.normalize(embeds, dim=1)  # 정규화
                image_embeddings.append(embeds.cpu().numpy())

        return np.concatenate(image_embeddings, axis=0)