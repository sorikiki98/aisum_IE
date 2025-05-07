import jax
import jax.numpy as jnp
import pickle
import numpy as np

from flax import serialization
from tqdm import tqdm

from magiclens.magiclens import MagicLensBackBone
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from image_embedding_model import ImageEmbeddingModel


class MagicLens(ImageEmbeddingModel):
    def __init__(self, model_name, cfg):
        super().__init__(model_name, cfg)

        tokenizer = clip_tokenizer.build_tokenizer()
        tokens = tokenizer("")  # used for index example
        tokens = np.array(tokens)
        self._tokens = tokens

        size = "large"
        model = MagicLensBackBone(size)
        rng = jax.random.PRNGKey(0)
        dummpy_input = {
            "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
            "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
        }
        params = model.init(rng, dummpy_input)
        model_dir = self.model_cfg["model_dir"]

        with open(f"{self.root_path}/{model_dir}/magic_lens_clip_{size}.pkl", "rb") as f:
            model_bytes = pickle.load(f)
        params = serialization.from_bytes(params, model_bytes)
        params = jax.device_put(params)
        self._params = params
        self._model = model

    @property
    def model(self):
        return self._model

    def forward(self, pil_images):
        processed_images = []
        for img in pil_images:
            ima = jnp.array(img)[jnp.newaxis, ...]
            ima = ima / (ima.max() + 1e-12)
            ima = jax.image.resize(ima, (1, 224, 224, 3), method='bilinear')
            processed_images.append(np.array(ima))
        iimages = jnp.concatenate(processed_images, axis=0)
        itokens = jnp.concatenate([self._tokens for _ in range(len(processed_images))], axis=0)
        iembeds = self.model.apply(self._params, {"ids": itokens, "image": iimages})[
            "multimodal_embed_norm"
        ]
        batch_embeddings_ndarray = jax.device_get(iembeds)
        batch_embeddings_ndarray = np.ascontiguousarray(batch_embeddings_ndarray, dtype=np.float32)

        return batch_embeddings_ndarray
