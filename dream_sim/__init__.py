from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch
import os
from dreamsim import dreamsim


class DreamSim(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)

        model, preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            use_patch_model=False,
            dreamsim_type="ensemble",
            cache_dir=str(os.path.join(self.root_path, self.model_cfg["model_dir"]))
        )

        self._model = model
        self._preprocess = preprocess

        self._model.eval()
        self._model.to(self.device)

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            tensor = self._preprocess(img)
            tensor = tensor.squeeze(0)
            processed_images.append(tensor)

        iimages = torch.stack(processed_images).to(self.device)

        with torch.no_grad():
            image_features = self.model.embed(iimages)

        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors

        batch_embeddings_ndarray = image_features.cpu().numpy()
        return batch_embeddings_ndarray
