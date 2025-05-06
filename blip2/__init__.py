from transformers import Blip2Model, Blip2Processor
from image_embedding_model import ImageEmbeddingModel
from tqdm import tqdm
import torch


class BLIP2(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)
        model = Blip2Model.from_pretrained(
            "Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16
        )
        preprocess = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xxl')

        model.to(self.device)
        model.eval()

        self._model = model
        self._preprocess = preprocess

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        prompt = "Question: Describe the product. Answer:"
        embedded_images = []
        with tqdm(total=len(pil_images), desc="Index examples") as progress:
            for img in pil_images:
                inputs = self._preprocess(images=img, text=prompt, return_tensors="pt").to(self.device, torch.float16)
                inputs["decoder_input_ids"] = inputs["input_ids"]
                outputs = self.model(**inputs)
                embedded_img = outputs['qformer_outputs']['pooler_output']
                embedded_img = embedded_img.squeeze(0)
                embedded_images.append(embedded_img)
                progress.update(1)

        image_features = torch.stack(embedded_images).to(self.device)
        batch_embeddings_ndarray = image_features.detach().cpu().numpy()

        return batch_embeddings_ndarray
