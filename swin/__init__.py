from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch
import timm
from timm import create_model
from timm.data import create_transform


class Swin(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        model = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        preprocess = create_transform(
            input_size=(3, 224, 224),
            is_training=False
        )
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
                processed_img = self._preprocess(img).to(self.device)
                processed_images.append(processed_img)
                progress.update(1)
        iimages = torch.stack(processed_images)
        batch_embeddings = self.model(iimages)
        batch_embeddings_ndarray = batch_embeddings.cpu().detach().numpy()

        return batch_embeddings_ndarray
