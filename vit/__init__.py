from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch
from transformers import ViTModel, ViTImageProcessor


class ViT(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        model.to(self.device)
        model.eval()
        preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

        self._model = model
        self._preprocess = preprocess

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            processed_img = self._preprocess(img, return_tensors="pt", input_data_format="channels_last")[
                "pixel_values"].to(self.device)
            processed_images.append(processed_img)
        iimages = torch.cat(processed_images, dim=0)
        outputs = self.model(iimages)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        batch_embeddings_ndarray = batch_embeddings.cpu().detach().numpy()

        return batch_embeddings_ndarray
