from model_utils import ImageEmbeddingModel
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor


class FashionCLIP(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        pretrained_path = "patrickjohncyh/fashion-clip"
        model = CLIPModel.from_pretrained(pretrained_path).to(self.device)
        preprocess = CLIPProcessor.from_pretrained(pretrained_path)

        model.to(self.device)
        model.eval()

        self._model = model
        self._preprocess = preprocess

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        with tqdm(total=len(pil_images), desc="Index examples") as progress:
            for img in pil_images:
                processed_img = self._preprocess(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
                processed_images.append(processed_img)
                progress.update(1)
        iimages = torch.stack(processed_images).to(self.device)
        image_features = self.model.get_image_features(iimages).detach().cpu()
        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors
        batch_embeddings_ndarray = image_features.numpy()

        return batch_embeddings_ndarray
