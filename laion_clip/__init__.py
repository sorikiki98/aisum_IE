from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch
import open_clip


class LaionCLIP(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device=self.device)
        model.to(self.device)
        model.eval()

        self._model = model
        self._preprocess = preprocess

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            processed_img = self._preprocess(img)
            processed_images.append(processed_img)
        iimages = torch.stack(processed_images).to(self.device)
        image_features = self.model.encode_image(iimages).detach().cpu()
        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors
        batch_embeddings_ndarray = image_features.numpy()

        return batch_embeddings_ndarray
