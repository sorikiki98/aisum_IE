from model_utils import ImageEmbeddingModel
from tqdm import tqdm
import torch
from torchvision import models


class EfficientNetV2(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        with tqdm(total=len(pil_images), desc="Index examples") as progress:
            for img in pil_images:
                processed_img = self._preprocess(img)
                processed_images.append(processed_img)
                progress.update(1)
        iimages = torch.stack(processed_images).to(self.device)
        batch_embeddings = self.model(iimages)
        batch_embeddings_ndarray = batch_embeddings.cpu().detach().numpy()

        return batch_embeddings_ndarray
