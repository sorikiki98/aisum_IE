from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch
import open_clip


class Coca(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        if model_name == "coca_laion2b":
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="coca_ViT-L-14",
                pretrained='laion2b_s13b_b90k',
                device=self.device
            )
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="coca_ViT-L-14",
                pretrained="mscoco_finetuned_laion2b_s13b_b90k",
                device=self.device
            )
        model.eval()
        self._model = model

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            processed_img = self._preprocess(img)
            processed_images.append(processed_img)
        iimages = torch.stack(processed_images).to(self.device)
        batch_embeddings = self.model.encode_image(iimages)
        batch_embeddings_ndarray = batch_embeddings.cpu().detach().numpy()

        return batch_embeddings_ndarray
