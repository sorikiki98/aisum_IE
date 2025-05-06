from image_embedding_model import ImageEmbeddingModel
import open_clip
import torch
from tqdm import tqdm


class Marqo(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        if model_name == "marqo_fashionclip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:Marqo/marqo-fashionCLIP', device=self.device)
        elif model_name == "marqo_fashionsiglip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:Marqo/marqo-fashionSigLIP', device=self.device)
        elif model_name == "marqo_ecommerce_b":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:Marqo/marqo-ecommerce-embeddings-B', device=self.device)
        elif model_name == "marqo_ecommerce_l":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:Marqo/marqo-ecommerce-embeddings-L', device=self.device)
        else:
            raise ValueError("Unknown marqo model name!")
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
                processed_img = self._preprocess(img)
                processed_images.append(processed_img)
                progress.update(1)

        iimages = torch.stack(processed_images).to(self.device)
        image_features = self.model.encode_image(iimages).detach().cpu()
        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors
        batch_embeddings_ndarray = image_features.numpy()

        return batch_embeddings_ndarray
