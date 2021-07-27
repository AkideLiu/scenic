"""Implementation of MLP-Mixer model."""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers


class MixerBlock(nn.Module):
  """Mixer block consisting of a token- and a channel-mixing phase.

  Attributes:
    channels_mlp_dim: Hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: Hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: The layer dropout rate (= stochastic depth).

  Returns:
    Output after mixer block.
  """
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  def get_stochastic_depth_mask(self, x: jnp.ndarray,
                                deterministic: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    if not deterministic and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.stochastic_depth, shape)
    else:
      return 0.0

  # Having this as a separate function makes it possible to capture the
  # intermediate representation via capture_intermediandarrates.
  def combine_branches(self, long_branch: jnp.ndarray,
                       short_branch: jnp.ndarray) -> jnp.ndarray:
    """Merges residual connections."""
    return long_branch + short_branch

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies the Mixer block to inputs."""
    if inputs.ndim != 3:
      raise ValueError('Input should be of shape `[batch, tokens, channels]`.')

    # Token mixing part, provides between-patches communication.
    x = nn.LayerNorm()(inputs)
    x = jnp.swapaxes(x, 1, 2)

    x = attention_layers.MlpBlock(
        mlp_dim=self.sequence_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_mixing')(
            x, deterministic=deterministic)

    x = jnp.swapaxes(x, 1, 2)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x *= 1.0 - self.get_stochastic_depth_mask(x, deterministic)
    x = self.combine_branches(x, inputs)

    # Channel-mixing part, which provides within-patch communication.
    y = nn.LayerNorm()(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.channels_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='channel_mixing')(
            y, deterministic=deterministic)

    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic)
    y *= 1.0 - self.get_stochastic_depth_mask(y, deterministic)
    return self.combine_branches(y, x)


class Mixer(nn.Module):
  """Mixer model.

  Attributes:
    num_classes: Number of output classes.
    patch_size: Patch size of the stem.
    hidden_size: Size of the hidden state of the output of model's stem.
    num_layers: Number of layers.
    channels_mlp_dim: hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: overall stochastic depth rate.
  """

  num_classes: int
  patch_size: Sequence[int]
  hidden_size: int
  num_layers: int
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:

    x = nn.Conv(
        self.hidden_size,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    for i in range(self.num_layers):
      p = (i / max(self.num_layers - 1, 1)) * self.stochastic_depth
      x = MixerBlock(
          channels_mlp_dim=self.channels_mlp_dim,
          sequence_mlp_dim=self.sequence_mlp_dim,
          dropout_rate=self.dropout_rate,
          stochastic_depth=p,
          name=f'mixerblock_{i}')(
              x, deterministic=not train)
    x = nn.LayerNorm(name='pre_logits_norm')(x)
    # Use global average pooling for classifier:
    x = jnp.mean(x, axis=1)
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)


class MixerMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Mixer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:
    return Mixer(
        num_classes=self.dataset_meta_data['num_classes'],
        patch_size=self.config.model.patch_size,
        hidden_size=self.config.model.hidden_size,
        num_layers=self.config.model.num_layers,
        channels_mlp_dim=self.config.model.channels_mlp_dim,
        sequence_mlp_dim=self.config.model.sequence_mlp_dim,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                patch_size=(4, 4),
                hidden_size=16,
                num_layers=1,
                channels_mlp_dim=32,
                sequence_mlp_dim=32,
                dropout_rate=0.,
                stochastic_depth=0,
            )
    })