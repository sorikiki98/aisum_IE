from .model import load, available_models
from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch


class Unicom(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        unicom_model, preprocess = load("ViT-B/32")
        unicom_model.to(self.device)
        unicom_model.eval()

        self._preprocess = preprocess
        self._model = unicom_model

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            processed_img = self._preprocess(img)
            processed_images.append(processed_img)
        iimages = torch.stack(processed_images).to(self.device)
        with torch.no_grad():
            batch_embeddings = self.model(iimages)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1,
                                                                        keepdim=True)
        batch_embeddings_ndarray = batch_embeddings.cpu().numpy()

        return batch_embeddings_ndarray
