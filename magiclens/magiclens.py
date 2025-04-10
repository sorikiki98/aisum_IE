from typing import Dict

from scenic.projects.baselines.clip import layers as clip_layers
from scenic.projects.baselines.clip import model as clip_model

import string
from typing import Optional, Tuple
import flax.linen as nn
from flax.linen import initializers
from flax.typing import Dtype
from flax.typing import Initializer

import jax
import jax.numpy as jnp
import numpy as np

default_kernel_init = initializers.lecun_normal()


class LayerNorm(nn.Module):
    """Layer norm used in Transformer layers."""

    dim: int = 1
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True

    def setup(self) -> None:
        if self.use_scale:
            self.scale = self.param('scale', jax.nn.initializers.ones, (self.dim,))
        if self.use_bias:
            self.bias = self.param('bias', jax.nn.initializers.zeros, (self.dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=[-1], keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=[-1], keepdims=True)
        normed_x = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        if self.use_scale:
            normed_x = normed_x * (1 + self.scale)
        if self.use_bias:
            normed_x = normed_x + self.bias

        return normed_x


class Weight(nn.Module):
    input_dim: int = 0
    hidden_dim: int = 0
    kernel_init: Initializer = default_kernel_init
    param_dtype: Dtype = jnp.float32

    def setup(self) -> None:
        self.w = self.param('w',
                            self.kernel_init,
                            (self.input_dim, self.hidden_dim),
                            self.param_dtype
                            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(x, self.w)


class Bias(nn.Module):
    hidden_dim: int = 0
    param_dtype: Dtype = jnp.float32

    def setup(self) -> None:
        self.b = self.param('b',
                            jax.nn.initializers.zeros,
                            (self.hidden_dim,),
                            self.param_dtype
                            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.b


class FFN(nn.Module):
    """Feed-forward network."""
    input_dim: int = 0
    output_dim: int = 0
    use_bias: bool = True
    use_relu: bool = True

    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init

    def setup(self) -> None:
        self.linear = Weight(self.input_dim, self.output_dim)
        if self.use_bias:
            self.bias = Bias(self.output_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear(x)
        if self.use_bias:
            x = self.bias(x)
        if self.use_relu:
            x = jax.nn.relu(x)
        return x


class TransformerFFN(nn.Module):
    """Feed-forward network used in Transformer layers with residual connection."""
    input_dim: int = 0
    output_dim: int = 0
    hidden_dim: int = 0
    use_bias: bool = True
    add_skip_connection: bool = True

    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init

    def setup(self) -> None:
        output_dim = self.output_dim
        if output_dim == 0:
            output_dim = self.input_dim
        self.ln = LayerNorm(dim=self.input_dim, name='layer_norm')
        self.ffn1 = FFN(input_dim=self.input_dim,
                        output_dim=self.hidden_dim,
                        use_bias=self.use_bias,
                        name='ffn_layer1')
        self.ffn2 = FFN(input_dim=self.hidden_dim,
                        output_dim=output_dim,
                        use_bias=self.use_bias,
                        use_relu=False,
                        name='ffn_layer2')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # pdb.set_trace()
        residual = x
        x = self.ln(x)
        x = self.ffn1(x)
        x = self.ffn2(x)
        if self.add_skip_connection:
            x = x + residual
        return x


class AttentionProjection(nn.Module):
    """Projection (e.g., k) used in self-attention.

    output_proj: Whether it is out projection or not. If False, we use
      "...D,DNH->...NH" for query,key,value projection. Otherwise we use
      "...NH,DNH->...D" for output projection.
    """
    input_dim: int = 0
    num_heads: int = 0
    dim_per_head: int = 0
    use_bias: bool = True
    output_proj: bool = False
    param_dtype: Dtype = jnp.float32

    def setup(self) -> None:
        hd_shape = [self.num_heads, self.dim_per_head]
        pc_shape = [self.input_dim] + hd_shape
        if self.output_proj:
            fan_in_axes, fan_out_axes = [-1], [-2, -3]
        else:
            fan_in_axes, fan_out_axes = [-3], [-1, -2]
        self.w = self.param(
            'w',
            jax.nn.initializers.lecun_normal(fan_in_axes, fan_out_axes),
            pc_shape,
        )
        if self.use_bias:
            if self.output_proj:
                self.b = self.param(
                    'b',
                    jax.nn.initializers.zeros,
                    (self.input_dim,),
                    dtype=self.param_dtype,
                )
            else:
                self.b = self.param(
                    'b', jax.nn.initializers.zeros, hd_shape, dtype=self.param_dtype
                )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('DHN')))
        shape = x.shape
        rank = len(shape)

        if self.output_proj:
            assert shape[-2:] == (self.num_heads, self.dim_per_head)
            batch_eqn = eqn_sym[: (rank - 2)]
            eqn = f'{batch_eqn}NH,DNH->{batch_eqn}D'
        else:
            assert (
                    shape[-1] == self.input_dim
            ), f'Expecting shape[-1] == p.input_dim, {shape[-1]} != {self.input_dim}'
            batch_eqn = eqn_sym[: (rank - 1)] if rank else '...'
            eqn = f'{batch_eqn}D,DNH->{batch_eqn}NH'
        ret = jnp.einsum(eqn, x, self.w)
        if self.use_bias:
            ret += self.b
        return ret


class PerDimScale(nn.Module):
    dim: int = 0
    param_dtype: Dtype = jnp.float32

    def setup(self) -> None:
        self.per_dim_scale = self.param(
            'per_dim_scale',
            jax.nn.initializers.ones,
            (self.dim,),
            dtype=self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.shape[-1] == self.dim
        r_softplus_0 = 1.442695041
        scale = jnp.array(r_softplus_0 / np.sqrt(self.dim), dtype=x.dtype)
        scale *= jax.nn.softplus(self.per_dim_scale)
        return x * scale


class DotProductAttention(nn.Module):
    """Self-attention used in Transformer layers."""
    input_dim: int = 0
    hidden_dim: int = 0
    num_heads: int = 1
    use_bias: bool = True
    dim_per_head: int = 0
    use_per_dim_scale: bool = False

    def setup(self) -> None:
        assert self.input_dim, 'input_dim is {}'.format(self.input_dim)
        assert self.hidden_dim, 'hidden_dim is {}'.format(self.hidden_dim)
        dim_per_head = self.dim_per_head
        if dim_per_head == 0:
            dim_per_head = self.hidden_dim // self.num_heads
            assert (
                    dim_per_head * self.num_heads == self.hidden_dim
            ), f'{dim_per_head} * {self.num_heads} != {self.hidden_dim}'

        self.key = AttentionProjection(input_dim=self.input_dim,
                                       num_heads=self.num_heads,
                                       dim_per_head=dim_per_head,
                                       use_bias=self.use_bias)
        self.query = AttentionProjection(input_dim=self.input_dim,
                                         num_heads=self.num_heads,
                                         dim_per_head=dim_per_head,
                                         use_bias=self.use_bias)
        self.value = AttentionProjection(input_dim=self.input_dim,
                                         num_heads=self.num_heads,
                                         dim_per_head=dim_per_head,
                                         use_bias=self.use_bias)

        if self.use_per_dim_scale:
            self.per_dim_scale = PerDimScale(dim=dim_per_head)

        self.post = AttentionProjection(self.input_dim,
                                        self.num_heads,
                                        dim_per_head,
                                        self.use_bias,
                                        output_proj=True)

    def _dot_atten(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Dot-product attention."""
        # query_vec: [B, T, D].
        # key_vec: [B, S, D].
        # value_vec: [B, S, D].

        # assert query.shape[:-1] == key.shape[:-1] == value.shape[:-1]
        if self.use_per_dim_scale:
            query = self.per_dim_scale(query)
        else:
            dim_per_head = self.hidden_dim // self.num_heads
            query *= dim_per_head ** -0.5

        logits = jnp.einsum('BTNH,BSNH->BNTS', query, key)
        # cap logits
        cap = jnp.array(50.0, dtype=logits.dtype)
        logits = cap * jnp.tanh(logits / cap)
        probs = jax.nn.softmax(logits, axis=-1).astype(key.dtype)
        encoded = jnp.einsum('BNTS,BSNH->BTNH', probs, value)

        return encoded, probs

    def __call__(self,
                 q_vector: jnp.ndarray,
                 k_vector: jnp.ndarray,
                 v_vector: jnp.ndarray,
                 atten_mask: None = None) -> Tuple[jnp.ndarray, jnp.ndarray]:

        query_proj = self.query(q_vector)
        key_proj = self.key(k_vector)
        value_proj = self.value(v_vector)
        encoded, atten_probs = self._dot_atten(query_proj, key_proj, value_proj)
        encoded = self.post(encoded)
        return encoded, atten_probs


class Transformer(nn.Module):
    """Transformer layer used in multimodal encoder."""
    num_heads: int
    # ff_layer, layer_norm, self_attention
    input_dim: int = 0
    hidden_dim: int = 0
    output_dim: int = 0
    use_bias: bool = True
    add_skip_connection: bool = True
    use_per_dim_scale: bool = False

    def setup(self) -> None:
        output_dim = self.output_dim
        if output_dim == 0:
            output_dim = self.input_dim
        self.ff_layer = TransformerFFN(
            self.input_dim,
            output_dim,
            self.hidden_dim,
            self.use_bias,
            self.add_skip_connection,
        )
        attn_hidden_dim = self.input_dim
        self.self_attention = DotProductAttention(
            input_dim=self.input_dim,
            hidden_dim=attn_hidden_dim,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            use_per_dim_scale=self.use_per_dim_scale,
        )
        self.layer_norm = LayerNorm(dim=self.input_dim, name='layer_norm')

    def __call__(
            self, x: jnp.ndarray, attn_mask=None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_normalized = self.layer_norm(x)
        atten_output, atten_probs = self.self_attention(
            x_normalized,
            x_normalized,
            x_normalized,
        )
        if self.add_skip_connection:
            atten_output += x
        output = self.ff_layer(atten_output)
        return output, atten_probs


class StackedTransformer(nn.Module):
    num_layers: int
    num_heads: int
    input_dim: int
    hidden_dim: int
    use_bias: bool = True
    add_skip_connection: bool = True
    use_per_dim_scale: bool = False

    def setup(self) -> None:
        assert self.num_layers > 0
        assert self.input_dim > 0
        assert self.hidden_dim > 0
        assert self.num_heads > 0
        output_dim = self.input_dim
        self.layers = [
            Transformer(num_heads=self.num_heads,
                        input_dim=self.input_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=output_dim,
                        use_bias=self.use_bias,
                        add_skip_connection=self.add_skip_connection,
                        use_per_dim_scale=self.use_per_dim_scale,
                        name=f'x_layers_{i}')
            for i in range(self.num_layers)
        ]

    def __call__(self, x: jnp.ndarray, attn_mask=None) -> jnp.ndarray:
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return x


class AttenTokenPoolingLayer(nn.Module):
    input_dim: int = 0
    query_dim: Optional[int] = None
    hidden_dim: int = 0
    num_heads: int = 1
    num_query_tokens: int = 1
    use_bias: bool = True
    use_per_dim_scale: bool = True
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init

    def setup(self):
        # Create sub-modules
        assert self.input_dim > 0, 'input_dim must be positive'
        query_dim = self.query_dim or self.input_dim
        ff_hidden_dim = (
            self.hidden_dim if self.hidden_dim > 0 else 4 * self.input_dim
        )
        self.pool_attn = DotProductAttention(
            input_dim=self.input_dim,
            hidden_dim=ff_hidden_dim,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            use_per_dim_scale=self.use_per_dim_scale,
            name='pool_attn',
        )

        self.pool_attn_ln = LayerNorm(
            dim=query_dim, epsilon=1e-6, name='pool_attn_ln'
        )
        self.pooling_attn_query = self.param(
            'pooling_attn_query',
            self.kernel_init,
            (self.num_query_tokens, query_dim),
            dtype=self.param_dtype,
        )

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        """Pooling layer.

        Args:
          embeds: (batch_size, seq_len, input_dim)
        Returns:
          pooled_output: (batch_size, query_dim)
        """
        batch_size, _ = embeds.shape[:2]
        query = jnp.tile(
            self.pooling_attn_query[jnp.newaxis, :, :], [batch_size, 1, 1]
        )
        key = embeds
        pooled_output, _ = self.pool_attn(query, key, embeds)
        pooled_output = self.pool_attn_ln(pooled_output)

        return pooled_output


MagicLensConfig = {
    'base': dict(
        embed_dim=512,
        ff_hidden_size=512 * 4,
        num_layers=4,
        num_heads=8,
        num_query_token=1,
        clip_model_name='vit_b16',
    ),
    'large': dict(
        embed_dim=768,
        ff_hidden_size=768 * 4,
        num_layers=4,
        num_heads=16,
        num_query_token=1,
        clip_model_name='vit_l14',
    ),
}


def largest_square_crop(images: jnp.ndarray) -> jnp.ndarray:
    assert images.ndim >= 4
    h, w, _ = images.shape[-3:]
    size = w if h > w else h

    pos_h = (h - w) // 2 if h > w else 0
    pos_w = (w - h) // 2 if w > h else 0

    return images[..., pos_h:pos_h + size, pos_w:pos_w + size, :]


class MagicLensBackBone(nn.Module):
    """MagicLens model built upon CLIP."""
    model_size: str = 'base'

    def setup(self):
        self.clip_model_name = MagicLensConfig[self.model_size]['clip_model_name']
        self.size: int = clip_model.IMAGE_RESOLUTION[self.clip_model_name]
        self.config: dict = clip_model.CONFIGS[self.clip_model_name]

        self.clip = clip_layers.CLIP(
            vocab_size=self.config['vocab_size'],
            embed_dim=self.config['embed_dim'],
            text_features=self.config['text_features'],
            text_num_layers=self.config['text_num_layers'],
            text_num_heads=self.config['text_num_heads'],
            vision_features=self.config['vision_features'],
            vision_num_layers=self.config['vision_num_layers'],
            vision_patch_size=self.config.get('vision_patch_size', None),
            vision_return_map=False,
            use_underscore_module_name=True,
            name='clip')

        self.multimodal_encoder = StackedTransformer(
            num_layers=MagicLensConfig[self.model_size]['num_layers'],
            num_heads=MagicLensConfig[self.model_size]['num_heads'],
            input_dim=MagicLensConfig[self.model_size]['embed_dim'],
            hidden_dim=MagicLensConfig[self.model_size]['ff_hidden_size'],
            use_bias=True,
            add_skip_connection=True,
            use_per_dim_scale=False,
            name='multimodal_encoder')

        self.contrastive_multimodal_pooler = AttenTokenPoolingLayer(
            input_dim=MagicLensConfig[self.model_size]['embed_dim'],
            query_dim=MagicLensConfig[self.model_size]['embed_dim'],
            num_heads=MagicLensConfig[self.model_size]['num_heads'],
            num_query_tokens=MagicLensConfig[self.model_size]['num_query_token'],
            use_bias=True,
            use_per_dim_scale=True,
            name='contrastive_multimodal_pooler')

    def _preprocess_images(self, images: jax.Array) -> jax.Array:
        """Center crop & resize image to be compatible with the underlied vision model."""
        assert images.ndim >= 4
        target_shape = images.shape[:-3] + (self.size, self.size, images.shape[-1])
        images = largest_square_crop(images)
        images = jax.image.resize(
            images, shape=target_shape, method='bilinear', antialias=True
        )
        # Apply CLIP-specific shifting/scaling.
        # The input to `normalize_image` is expected to be in [0, 1].
        images = clip_model.normalize_image(images)
        return images

    def clip_encode(self, input_batch: Dict) -> Tuple[jax.Array, jax.Array]:
        """Computes CLIP embeds for the given batch of images and texts.

        Args:
          input_batch: A Dict of the following fields:
            * ids: [B, T] or [B, 1, T]. Text token ids
            * paddings: [B, T] or [B, 1, T]. Text token paddings.
            * image: [B, H, W, 3]. Input image.

        Returns:
          image_embs: [B, D]
          text_embs: [B, D]
          patch_embeds: [B, N, D]
          token_embds: [B, T, D]
        """
        assert input_batch['ids'].ndim <= 3
        if input_batch['ids'].ndim == 3:
            # Only takes the first caption.
            input_batch['ids'] = input_batch['ids'][:, 0, :]
        images = self._preprocess_images(input_batch['image'])
        image_embs, text_embs = self.clip(
            images, input_batch['ids'], normalize=False
        )
        return image_embs, text_embs

    def _normalize_embed(self, embed: jax.Array) -> jax.Array:
        """Applies normalization on the input embedding.

        Args:
          embed: [B, D]. The input embedding to normalize.

        Returns:
          The normalized embedding.
        """
        # Always converts embed to float32 for all precisions.
        embed = jnp.asarray(embed, dtype=jnp.float32)
        # return py_utils.l2_normalize(embed, axis=-1)
        norm = jnp.sqrt(jnp.sum(embed * embed, axis=-1, keepdims=True) + 1e-12)
        return embed / norm

    def __call__(self, input_batch: Dict) -> Dict:
        """Computes the multimodal embeddings.

        It computes the multimodal embeddings pooling from both
        text embeddings and image *generative* embeddings.
        If text is empty, use image pooling only.

        Args:
          input_batch: A Dict of the following fields:
            * ids: [B, T] or [B, 1, T]. Text token ids
            * paddings: [B, T] or [B, 1, T]. Text token paddings.
            * image: [B, H, W, 3]. Input image.
        Returns:
          A Dict contains the following fields:
            * multimodal_embed: [B, D], multimodal embedding
            * multimodal_embed_norm: [B, D], normalized multimodal embedding
        """
        img_embed, txt_embed = self.clip_encode(input_batch)  # [B, D], [B, D]
        img_embed = img_embed.reshape(
            [-1, 1, img_embed.shape[-1]])  # [B, 1, D]
        txt_embed = txt_embed.reshape(
            [-1, 1, txt_embed.shape[-1]])  # [B, 1, D]

        concate_mm_embed = jnp.concatenate([img_embed, txt_embed], axis=1)

        multimodal_embed = self.multimodal_encoder(  # [B, 2, D]
            concate_mm_embed
        )

        multimodal_embed = self.contrastive_multimodal_pooler(
            multimodal_embed)
        multimodal_embed = multimodal_embed[:, 0]

        multimodal_embed_norm = self._normalize_embed(multimodal_embed)

        # placeholder for model matching
        # contrastive_loss = 0.0
        return {
            'multimodal_embed': multimodal_embed,
            'multimodal_embed_norm': multimodal_embed_norm,
        }
