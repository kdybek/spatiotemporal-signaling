import einops
import numpy as np
import jax

import dataclasses
import re
from typing import Any
import jax.numpy as jnp
from flax import linen as nn


class PatchEmbedding(nn.Module):
    """Extracts patches with a single learned linear projection."""
    patch_size: list[int]
    num_features: int

    @nn.compact
    def __call__(self, images):
        return nn.Conv(features=self.num_features, kernel_size=self.patch_size, strides=self.patch_size, padding='VALID')(images)


def get_mae_sinusoid_encoding_table(n_position, d_hid, dtype=jnp.float32):
    """Sinusoid positional encoding table for MAE."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return jnp.asarray(sinusoid_table, dtype)[None, ...]


class SincosPosEmb(nn.Module):
    """Returns sinusoidal positional embedding given the shape of the tokens."""
    base_token_shape: list[int] | None = None

    @nn.compact
    def __call__(self, tokens_shape):
        d = tokens_shape[-1]
        if self.base_token_shape is not None:
            h, w = self.base_token_shape
        else:
            h, w = tokens_shape[-3], tokens_shape[-2]

        posenc = get_mae_sinusoid_encoding_table(np.prod((h, w)), d)
        posenc = einops.rearrange(posenc, '... (h w) d -> ... h w d', h=h, w=w)
        *b, tokens_h, tokens_w, _ = tokens_shape
        for _ in range(len(b)-1):
            posenc = jnp.expand_dims(posenc, axis=0)
        if tokens_h != h or tokens_w != w:
            posenc = jax.image.resize(
                posenc, (*b, tokens_h, tokens_w, d), method='bicubic')

        return posenc


class SincosPosEmb1D(nn.Module):
    """Returns sinusoidal positional embedding given the shape of the tokens."""
    base_token_length: int | None = None

    @nn.compact
    def __call__(self, tokens_shape):
        d = tokens_shape[-1]
        if self.base_token_length is not None:
            length = self.base_token_length
        else:
            length = tokens_shape[-2]

        posenc = get_mae_sinusoid_encoding_table(length, d)
        *b, tokens_length, _ = tokens_shape
        for _ in range(len(b)-1):
            posenc = jnp.expand_dims(posenc, axis=0)
        if tokens_length != length:
            posenc = jax.image.resize(
                posenc, (*b, tokens_length, d), method='linear')

        return posenc


class Tokenizer(nn.Module):
    """Simple tokenizer."""
    patch_embedding: nn.Module
    posenc: nn.Module

    @nn.compact
    def __call__(self, image):
        tokens = self.patch_embedding(image)
        posenc = self.posenc(tokens.shape)
        tokens += posenc
        return tokens


class TransformerMLP(nn.Module):
    """Simple MLP with a single hidden layer for use in Transformer blocks."""
    hidden_size: int = None  # Defaults to 4 times input dims.

    @nn.compact
    def __call__(self, inputs):
        d = inputs.shape[-1]
        hidden_size = 4 * d if self.hidden_size is None else self.hidden_size
        h = nn.Dense(
            features=hidden_size,
            name='dense_in',
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=inputs.dtype,
        )(inputs)
        h = nn.gelu(h)
        return nn.Dense(
            features=inputs.shape[-1],
            name='dense_out',
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            dtype=h.dtype,
        )(h)


class PreNormBlock(nn.Module):
    """Pre-LN Transformer layer (default transformer layer)."""

    attention: Any
    mlp: nn.Module
    attention_norm: Any
    mlp_norm: Any

    @nn.compact
    def __call__(self, tokens):
        norm_tokens = self.attention_norm(tokens)
        tokens += self.attention(
            inputs_q=norm_tokens,
            inputs_k=norm_tokens,
            inputs_v=norm_tokens,
        )
        norm_tokens = self.mlp_norm(tokens)
        return tokens + self.mlp(norm_tokens)


VIT_SIZES = {
    'mu': (32, 1, 128, 2),
    'Ti': (192, 12, 768, 3),
    'S': (384, 12, 1536, 6),
    'M': (512, 12, 2048, 8),
    'B': (768, 12, 3072, 12),
    'L': (1024, 24, 4096, 16),
    'H': (1280, 32, 5120, 16),
    'g': (1408, 40, 6144, 16),
    'G': (1664, 48, 8192, 16),
    'e': (1792, 56, 15360, 16),
    'S_temp': (384, 4, 1536, 6),
    'M_temp': (512, 4, 2048, 8),
    'B_temp': (768, 4, 3072, 12),
    'L_temp': (1024, 8, 4096, 16),
}


@dataclasses.dataclass(frozen=True)
class ViTSpec:
    """Spec for the size of a Vision Transformer."""

    hidden_size: int  # Dimension of tokens passed between blocks.
    num_layers: int  # Number of trasformer blocks.
    mlp_size: int  # Hidden dimension of the MLP in each block.
    num_heads: int  # Number of attention heads.
    patch_size: int = None  # Patch size of initial image patches.

    @classmethod
    def from_variant_string(cls, variant_str: str):
        """Parse variant strings like "ViT-L", "B", or "Ti/16"."""
        r = re.match(
            r'^([Vv][Ii][Tt][-_])?(?P<name>[a-zA-Z]{1,2})(/(?P<patch>\d+))?$',
            variant_str,
        )
        if r is None:
            raise ValueError(f'Invalid variant string: {variant_str!r}.')
        name = r.groupdict()['name']
        spec = cls(*VIT_SIZES[name])

        patch_size = r.groupdict()['patch']
        if patch_size is not None:
            spec = dataclasses.replace(spec, patch_size=int(patch_size))
        return spec

    @property
    def kwargs(self):
        kwargs = dict(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            mlp_size=self.mlp_size,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
        )
        if self.patch_size is None:
            del kwargs['patch_size']
        return kwargs


class Transformer(nn.Module):
    """Simple transformer model."""
    layers: tuple[Any]

    @nn.compact
    def __call__(self, tokens):
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = nn.LayerNorm(dtype=tokens.dtype)(tokens)
        return tokens

    @classmethod
    def from_variant_str(cls, variant_str: str, **kwargs):
        vit_spec = ViTSpec.from_variant_string(variant_str)
        all_kwargs = vit_spec.kwargs | kwargs
        all_kwargs.pop('patch_size', None)
        all_kwargs.pop('hidden_size', None)
        return cls.from_spec(**all_kwargs)

    @classmethod
    def from_spec(
        cls,
        num_heads: int,
        num_layers: int,
        mlp_size=None,
        dtype=jnp.float32,
        qk_features=None,
        v_features=None,
        **kwargs,
    ):
        return cls(
            layers=tuple(
                PreNormBlock(
                    attention_norm=nn.LayerNorm(dtype=dtype),
                    mlp_norm=nn.LayerNorm(dtype=dtype),
                    attention=ImprovedMultiHeadDotProductAttention(
                        num_heads=num_heads,
                        qk_features=qk_features,
                        v_features=v_features,
                    ),
                    mlp=TransformerMLP(hidden_size=mlp_size),
                )
                for _ in range(num_layers)
            ),
            **kwargs,
        )


def softmax(x):
    return jax.nn.softmax(x.astype(jnp.float32), axis=-1).astype(jnp.float32)


def dot_product_attention_weights(query, key):
    query = query / jnp.sqrt(query.shape[-1])
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
    return softmax(attn_weights)


class ImprovedMultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention."""

    num_heads: int
    qk_features: int = None
    v_features: int = None
    out_features: int = None

    @nn.compact
    def __call__(
        self,
        inputs_q,
        inputs_k=None,
        inputs_v=None,
        *,
        bias=None,
        mask=None,
    ):
        qk_features = self.qk_features or inputs_q.shape[-1]
        v_features = self.v_features or qk_features

        if inputs_k is None:
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        def dense(name, x, features):
            return nn.DenseGeneral(
                features=(self.num_heads, features // self.num_heads),
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=nn.initializers.zeros_init(),
                use_bias=True,
                dtype=x.dtype,
                name=name,
            )(x)

        query = dense('query', inputs_q, qk_features)
        key = dense('key', inputs_k, qk_features)
        value = dense('value', inputs_v, v_features)

        # Compute attention weights.
        attn_weights = dot_product_attention_weights(query=query, key=key)

        # Return weighted sum over values for each query position.
        x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

        # Back to the original input dimensions.
        return nn.DenseGeneral(
            features=self.out_features or inputs_q.shape[-1],
            axis=(-2, -1),
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros_init(),
            use_bias=True,
            dtype=x.dtype,
            name='out',
        )(x)


class CrossAttentionTransformer(nn.Module):
    """Cross attention transformer."""
    num_heads: int
    num_layers: int
    num_feats: int
    mlp_dim: int
    dtype: Any

    def setup(self):
        self.xa_blocks = [CrossAttentionBlock(
            num_heads=self.num_heads, num_feats=self.num_feats,
            mlp_dim=self.mlp_dim, dtype=self.dtype,
        ) for _ in range(self.num_layers)]
        self.output_norm = nn.LayerNorm(dtype=self.dtype)

    def __call__(self, inputs, inputs_kv):
        for i in range(self.num_layers):
            inputs = self.xa_blocks[i](inputs, inputs_kv)
        return self.output_norm(inputs)


class CrossAttentionBlock(nn.Module):
    """Cross attention block."""

    num_heads: int
    num_feats: int
    mlp_dim: int
    dtype: Any

    def setup(self):
        self.attention_norm = nn.LayerNorm(dtype=self.dtype)
        self.mlp_norm = nn.LayerNorm(dtype=self.dtype)
        self.ca_attention_norm = nn.LayerNorm(dtype=self.dtype)
        self.attention = ImprovedMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qk_features=self.num_feats,
            v_features=self.num_feats,
        )
        self.ca_attention = ImprovedMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qk_features=self.num_feats,
            v_features=self.num_feats,
        )
        self.mlp = TransformerMLP(hidden_size=self.mlp_dim)

    def __call__(self, inputs, inputs_kv):
        x = inputs
        x = x + self.ca_attention(inputs_q=self.ca_attention_norm(x),
                                  inputs_k=inputs_kv, inputs_v=inputs_kv)
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.attention(self.attention_norm(x))
        return x


class Detokenizer(nn.Module):
    """Detokenize tokens to pixel-space patches via a learned linear projection."""
    patch_size: tuple[int, int]
    num_features: int = 3

    @nn.compact
    def __call__(self, x):
        num_total_features = self.num_features * np.prod(self.patch_size)
        x = nn.Dense(num_total_features, dtype=x.dtype)(x)
        # Rearrange from (... H W (h w C)) to (... H*h W*w C)
        x = einops.rearrange(
            x,
            '... H W (h w C) -> ... (H h) (W w) C',
            h=self.patch_size[0],
            w=self.patch_size[1],
            C=self.num_features,
        )
        return x


def random_masking(rng_key, tokens, mask_ratio):
    """Random masking: shuffle tokens and return visible/masked split info.

    Args:
      rng_key: JAX random key.
      tokens: Input tokens of shape (..., N, D).
      mask_ratio: Fraction of tokens to mask (drop).

    Returns:
      visible_tokens: The unmasked subset of tokens.
      inds_restore: Indices to restore the original token order.
      mask: Binary mask (1 = masked, 0 = visible) in original order.
    """
    n_tokens = tokens.shape[-2]
    n_keep = int(n_tokens * (1.0 - mask_ratio))

    inds = jnp.arange(n_tokens)
    inds = jnp.broadcast_to(inds, tokens.shape[:-1])[..., jnp.newaxis]
    inds = jax.random.permutation(rng_key, inds, axis=-2)
    inds_restore = jnp.argsort(inds, axis=-2)

    shuffled_tokens = jnp.take_along_axis(tokens, inds, axis=-2)
    visible_tokens = shuffled_tokens[..., :n_keep, :]

    mask = jnp.concatenate([
        jnp.zeros(tokens.shape[:-2] + (n_keep, 1)),
        jnp.ones(tokens.shape[:-2] + (n_tokens - n_keep, 1)),
    ], axis=-2)
    mask = jnp.take_along_axis(mask, inds_restore, axis=-2)

    return visible_tokens, inds_restore, mask


class AttentionPooler(nn.Module):
    """Attention-based pooling layer."""
    num_heads: int
    num_feats: int

    @nn.compact
    def __call__(self, tokens):
        *b, N, D = tokens.shape
        query = self.param('query', nn.initializers.normal(stddev=0.02), (1, 1, D))
        query = jnp.broadcast_to(query, (*b, 1, D))
        attn = ImprovedMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qk_features=self.num_feats,
            v_features=self.num_feats,
        )
        pooled = attn(inputs_q=query, inputs_k=tokens, inputs_v=tokens)
        return pooled[..., 0, :]


class MoCoMAE(nn.Module):
    """Motion-content masked autoencoder model."""

    # Encoder
    tokenizer: nn.Module
    spatial_encoder: nn.Module
    temporal_encoder: nn.Module
    temporal_posenc: nn.Module
    latent_emb_dim: int

    # Decoder
    decoder: nn.Module
    decoder_embedder: nn.Module
    delta_embedder: nn.Module
    latent_posenc: nn.Module
    detokenizer: nn.Module
    decoder_emb_dim: int
    masking_ratio: float
    content_pooling: str  # 'mean' or 'random'
    latent_decomposition: str  # 'residual' or 'concat'

    def setup(self):
        assert self.latent_decomposition in ['residual', 'concat'], (
            f'Invalid latent_decomposition: {
                self.latent_decomposition}. Must be "residual" or "concat".'
        )

        self.motion_token = self.param('motion_token', nn.initializers.normal(
            stddev=0.02), (1, self.latent_emb_dim))
        if self.decoder is not None:
            self.mask_token = self.param('mask_token', nn.initializers.normal(
                stddev=0.02), (1, self.decoder_emb_dim))

    @nn.compact
    def __call__(
        self,
        source_frames,
        target_frames,
        target_deltas=None,
        rng_key=None,
    ):
        """Full forward pass with encoder + decoder for masked reconstruction.

        Args:
          source_frames: Source (context) frames, shape (B, T, H, W, C).
          target_frames: Target frames to reconstruct, shape (B, TT, H, W, C).
          target_deltas: Optional temporal deltas, shape (B, TT), integer.
          rng_key: JAX random key for masking.

        Returns:
          A dictionary containing:
            - 'reconstructed': Reconstructed target frames, shape (B, TT, H, W, C).
            - 'mask': Binary mask indicating which tokens were masked, shape (B, TT, h, w, 1).
            - 'features': Latent features from the encoder, shape (B, N, D).
            - 'content_features': Content latent features from the encoder, shape (B, N, D).
            - 'motion_features': Motion latent features from the encoder, shape (B, N, D).
        """
        if rng_key is None:
            rng_key = self.make_rng('default')

        # Tokenize source and target frames
        source_tokens = self.tokenizer(source_frames)
        _, num_source_frames, source_tokens_h, source_tokens_w, source_tokens_d = (
            source_tokens.shape
        )
        target_tokens = self.tokenizer(target_frames)
        B, num_target_frames, target_tokens_h, target_tokens_w, target_tokens_d = (
            target_tokens.shape
        )
        N = source_tokens_h * source_tokens_w

        # Flatten source tokens
        source_tokens = einops.rearrange(source_tokens, 'B T h w D -> (B T) (h w) D')

        # Encode source tokens
        encoded_source_tokens = self.spatial_encoder(source_tokens)

        # Pool content latents from encoded source tokens
        patch_tokens = einops.rearrange(
            encoded_source_tokens, '(B T) N D -> B N T D', B=B, T=num_source_frames)

        if self.content_pooling == 'mean':
            content_latents = jnp.mean(patch_tokens, axis=-2)
        elif self.content_pooling == 'random':
            rng_key, subkey = jax.random.split(rng_key)
            rand_idx = jax.random.randint(
                subkey, shape=(B,), minval=0, maxval=num_source_frames)
            content_latents = jnp.take_along_axis(
                patch_tokens,
                rand_idx[..., jnp.newaxis, jnp.newaxis, jnp.newaxis],
                axis=-2
            )
            content_latents = jnp.squeeze(content_latents, axis=-2)
        else:
            raise ValueError(f'Invalid content_pooling: {
                             self.content_pooling}. Must be "mean" or "random".')

        # Add temporal positional encoding to patch tokens
        temporal_posenc = self.temporal_posenc(patch_tokens.shape)
        patch_tokens += temporal_posenc

        # Add motion token and encode through temporal encoder
        motion_token_t = jnp.broadcast_to(
            self.motion_token,
            (B, N, 1, self.motion_token.shape[-1])
        )
        motion_inputs = jnp.concatenate([motion_token_t, patch_tokens], axis=-2)
        motion_latents = self.temporal_encoder(motion_inputs)
        motion_latents = motion_latents[..., 0, :]

        if self.latent_decomposition == 'residual':
            latents = content_latents + motion_latents
        elif self.latent_decomposition == 'concat':
            latents = jnp.concatenate([content_latents, motion_latents], axis=-1)

        # Mask target tokens
        target_tokens_flat = einops.rearrange(
            target_tokens, '... h w D -> ... (h w) D'
        )
        visible_target, inds_restore, mask = random_masking(
            rng_key, target_tokens_flat, self.masking_ratio
        )

        # Encode target frames
        visible_target = einops.rearrange(
            visible_target, 'B T N D -> (B T) N D'
        )
        encoded_targets = self.spatial_encoder(visible_target)
        encoded_targets = einops.rearrange(
            encoded_targets, '(B T) N D -> B T N D', B=B, T=num_target_frames
        )

        # Embed target tokens for decoder
        embedded_target_tokens = self.decoder_embedder(encoded_targets)

        # Concat unmasked tokens with mask tokens and restore order
        mask_tokens = jnp.broadcast_to(
            self.mask_token,
            [
                B,
                num_target_frames,
                inds_restore.shape[-2] - embedded_target_tokens.shape[-2],
                self.mask_token.shape[-1],
            ],
        )
        shuffled_tokens = jnp.concatenate(
            [embedded_target_tokens, mask_tokens], axis=-2)
        unshuffled_tokens = jnp.take_along_axis(
            shuffled_tokens, inds_restore, axis=-2
        )

        # Encode target deltas
        if self.delta_embedder is not None and target_deltas is not None:
            target_deltas_onehot = jax.nn.one_hot(target_deltas, 64, axis=-1)
            delta_tokens = self.delta_embedder(
                target_deltas_onehot)[..., jnp.newaxis, :]
        else:
            delta_tokens = None

        # Build positional embedding for decoder tokens
        latent_posenc_shape = (
            1, target_tokens_h, target_tokens_w,
            embedded_target_tokens.shape[-1],
        )
        latent_posenc = self.latent_posenc(latent_posenc_shape)
        latent_posenc = jnp.reshape(
            latent_posenc,
            [1, 1, target_tokens_h * target_tokens_w, latent_posenc.shape[-1]],
        )
        latent_posenc = jnp.broadcast_to(latent_posenc, unshuffled_tokens.shape)

        if delta_tokens is not None:
            unshuffled_tokens += delta_tokens
        unshuffled_tokens += latent_posenc
        to_decode = unshuffled_tokens

        # Prepare KV from encoded source tokens
        inputs_kv = self.decoder_embedder(latents)
        inputs_kv = inputs_kv[..., jnp.newaxis, :, :]
        inputs_kv = jnp.tile(inputs_kv, [num_target_frames, 1, 1])

        # Decode
        decoded = self.decoder(to_decode, inputs_kv=inputs_kv)

        # Reshape back to image space
        reconstructed = jnp.reshape(
            decoded,
            [
                B,
                num_target_frames,
                target_tokens_h,
                target_tokens_w,
                decoded.shape[-1],
            ],
        )
        mask = jnp.reshape(
            mask, [B, num_target_frames, target_tokens_h, target_tokens_w, 1]
        )

        # Detokenize to pixel space
        reconstructed = self.detokenizer(reconstructed)

        return {
            'reconstructed': reconstructed,
            'mask': mask,
            'features': latents,
            'content_features': content_latents,
            'motion_features': motion_latents,
        }


def get_mocomae(num_channels, masking_ratio, variant, content_pooling, latent_decomposition):
    if variant == 'L':
        spatial_encoder_variant = 'L'
        temporal_encoder_variant = 'L_temp'
        embed_dim = 1024
    elif variant == 'B':
        spatial_encoder_variant = 'B'
        temporal_encoder_variant = 'B_temp'
        embed_dim = 768
    elif variant == 'M':
        spatial_encoder_variant = 'M'
        temporal_encoder_variant = 'M_temp'
        embed_dim = 512
    elif variant == 'S':
        spatial_encoder_variant = 'S'
        temporal_encoder_variant = 'S_temp'
        embed_dim = 384
    else:
        raise ValueError(f'Unknown variant: {variant}')

    return MoCoMAE(
        tokenizer=Tokenizer(
            patch_embedding=PatchEmbedding(
                patch_size=[1, 16, 16], num_features=embed_dim),
            posenc=SincosPosEmb(),
        ),
        spatial_encoder=Transformer.from_variant_str(
            variant_str=spatial_encoder_variant, dtype=jax.numpy.bfloat16),
        temporal_encoder=Transformer.from_variant_str(
            variant_str=temporal_encoder_variant, dtype=jax.numpy.bfloat16),
        temporal_posenc=SincosPosEmb1D(),
        latent_emb_dim=embed_dim,
        # Decoder components
        decoder=CrossAttentionTransformer(
            num_layers=8,
            num_heads=16,
            num_feats=512,
            mlp_dim=2048,
            dtype=jax.numpy.bfloat16,
        ),
        decoder_embedder=nn.Dense(512),
        delta_embedder=nn.Dense(512),
        latent_posenc=SincosPosEmb(),
        detokenizer=Detokenizer(patch_size=(16, 16), num_features=num_channels),
        decoder_emb_dim=512,
        masking_ratio=masking_ratio,
        content_pooling=content_pooling,
        latent_decomposition=latent_decomposition
    )
