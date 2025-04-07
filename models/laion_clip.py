import torch
import open_clip


class LaionCLIP:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device=device)
        self._device = device

    def embed_images(self, image_batch):
        image_features = self._model.encode_image(image_batch).detach().cpu()  # [bs, 1024]
        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors
        image_features.to(self._device)

        return image_features

    def get_embedding_dimension(self):
        return self._model.visual.output_dim
