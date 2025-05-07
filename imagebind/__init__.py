import os
import torch
from torchvision import transforms
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from image_embedding_model import ImageEmbeddingModel


class ImageBind(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        weight_path = os.path.join(self.root_path, self.model_cfg["model_dir"])
        model = imagebind_model.imagebind_huge(pretrained=True, weight_folder_path=weight_path)
        model.to(self.device)
        model.eval()
        self._model = model
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            processed_img = self._preprocess(img)
            processed_images.append(processed_img)
        iimages = torch.stack(processed_images).to(self.device)
        inputs = {ModalityType.VISION: iimages}
        with torch.no_grad():
            batch_embeddings = self.model(inputs)
            batch_embeddings = batch_embeddings[ModalityType.VISION]
        batch_embeddings_ndarray = batch_embeddings.cpu().numpy()

        return batch_embeddings_ndarray
