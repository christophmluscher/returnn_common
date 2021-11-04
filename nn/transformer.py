"""
Transformer Modules
"""

from __future__ import annotations
from typing import Optional, Any, Union
from . import relu, layer_norm, dropout, LayerRef, Module, ModuleList, Linear


class TransformerEncoderLayer(Module):
  """
  Defines one layer of a standard transformer encoder
  """
  def __init__(self, dim_model: int, num_heads: int, dim_ff: int = 2048, drop: float = 0.1, activation=relu,
               layer_norm_eps: float = 1e-5, norm_first: bool = False) -> None:
    """
    :param dim_model: hidden dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param drop: Dropout value, PyTorch name: dropout
    :param activation: activation functional
    :param layer_norm_eps: Epsilon value for layer normalization
    :param norm_first: Whether to do layer norm before or afterwards
    """
    super().__init__()
    self.self_attn = MultiheadAttention(dim_model, num_heads, dropout=dropout)  # will change with Attention Modules

    self.linear1 = Linear(dim_ff)
    self.linear2 = Linear(dim_model)

    self.activationivation = activation
    self.norm_first = norm_first
    self.norm_eps = layer_norm_eps
    self.dropout = drop

  def forward(self, inp: LayerRef) -> LayerRef:
    """
    Two possible forward variants of encoder, defined by self.norm_first
    """
    x = inp
    if self.norm_first:
      x = x + self._sa_block(layer_norm(x, epsilon=self.norm_eps))
      x = x + self._ff_block(layer_norm(x, epsilon=self.norm_eps))
    else:
      x = layer_norm(x + self._sa_block(x), epsilon=self.norm_eps)
      x = layer_norm(x + self._ff_block(x), epsilon=self.norm_eps)

    return x

  # self-attention block
  def _sa_block(self, x: LayerRef) -> LayerRef:
    x = self.self_attn(x, x, x)
    return dropout(x, self.dropout)

  # feed forward block
  def _ff_block(self, x: LayerRef) -> LayerRef:
    x = self.linear2(dropout(self.activationivation(self.linear1(x)), self.dropout))
    return dropout(x, self.dropout)


class TransformerEncoder(Module):
  """
  Defines the full Encoder of the standard transformer
  """
  def __init__(self, encoder_layer: Union[TransformerEncoderLayer, Any], num_layers: int, norm=None,
               layer_norm_eps: float = 1e-5):
    """
    :param encoder_layer: Encoder layer to be stacked num_layers times
    :param num_layers: Number of layers
    :param norm: normalization functional
    :param layer_norm_eps: Epsilon value for layer normalization
    """
    super().__init__()
    import copy
    self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    self.num_layers = num_layers
    self.norm = norm
    self.norm_eps = layer_norm_eps

  def forward(self, inp: LayerRef) -> LayerRef:
    """
    Executes the encoder layer as often as in num_layers defined
    """
    output = inp
    for mod in self.layers:
      output = mod(output)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class TransformerDecoderLayer(Module):
  """
  Defines one layer of a standard transformer decoder
  """
  def __init__(self, dim_model: int, num_heads: int, dim_ff: int = 2048, drop: float = 0.1, activation=relu,
               layer_norm_eps: float = 1e-5, norm_first: bool = False):
    """
    :param dim_model: hidden dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param drop: Dropout value, PyTorch name: dropout
    :param activation: activation functional
    :param layer_norm_eps: Epsilon value for layer normalization
    :param norm_first: Whether to do layer norm before or afterwards
    """
    super().__init__()
    self.dropout = drop
    self.self_attn = MultiheadAttention(dim_model, num_heads, dropout=self.dropout)  # will change with AttentionModules
    self.multihead_attn = MultiheadAttention(dim_model, num_heads, dropout=self.dropout)  # will change with AM

    self.linear1 = Linear(dim_ff)
    self.linear2 = Linear(dim_model)

    self.norm_first = norm_first
    self.norm_eps = layer_norm_eps
    self.activationivation = activation

  def forward(self, tgt: LayerRef, memory: LayerRef) -> LayerRef:
    """
    Two possible forward variants of decoder, defined by self.norm_first
    """
    x = tgt
    if self.norm_first:
      x = x + self._sa_block(layer_norm(x, epsilon=self.norm_eps))
      x = x + self._mha_block(layer_norm(x, epsilon=self.norm_eps), memory)
      x = x + self._ff_block(layer_norm(x, epsilon=self.norm_eps))
    else:
      x = layer_norm(x + self._sa_block(x), epsilon=self.norm_eps)
      x = layer_norm(x + self._mha_block(x, memory), epsilon=self.norm_eps)
      x = layer_norm(x + self._ff_block(x), epsilon=self.norm_eps)

    return x

  # self-attention block
  def _sa_block(self, x: LayerRef) -> LayerRef:
    x = self.self_attn(x, x, x)  # will change with Attention Modules
    return dropout(x, self.dropout)

  # multihead attention block
  def _mha_block(self, x: LayerRef, mem: LayerRef) -> LayerRef:
    x = self.multihead_attn(x, mem, mem)  # will change with Attention Modules
    return dropout(x, self.dropout)

  # feed forward block
  def _ff_block(self, x: LayerRef) -> LayerRef:
    x = self.linear2(dropout(self.activationivation(self.linear1(x)), self.dropout))
    return dropout(x, self.dropout)


class TransformerDecoder(Module):
  """
  Defines the full Decoder of the standard transformer
  """
  def __init__(self, decoder_layer: Union[TransformerDecoderLayer, Any], num_layers: int, norm=None,
               layer_norm_eps: float = 1e-5):
      """
      :param decoder_layer: Decoder layer to be stacked num_layers times
      :param num_layers: Number of layers
      :param norm: normalization functional
      :param layer_norm_eps: Epsilon value for layer normalization
      """
      super(TransformerDecoder, self).__init__()
      import copy
      self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
      self.num_layers = num_layers
      self.norm = norm
      self.norm_eps = layer_norm_eps

  def forward(self, tgt: LayerRef, memory: LayerRef) -> LayerRef:
    """
    Executes the decoder layer as often as in num_layers defined
    """
    output = tgt
    for mod in self.layers:
      output = mod(output, memory)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class Transformer(Module):
  """
  Standard Transformer Module
  """
  def __init__(self, dim_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6,
               num_decoder_layers: int = 6, dim_ff: int = 2048, drop: float = 0.1,
               activation=relu, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
               layer_norm_eps: float = 1e-5) -> None:
    """
    :param dim_model: hidden dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param num_encoder_layers: Number of encoder layers
    :param num_decoder_layers: Number of decoder layers
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param drop: Dropout value, PyTorch name: dropout
    :param activation: activation functional
    :param custom_encoder: Custom Encoder layer to replace the standard layer
    :param custom_decoder: Custom Decoder layer to replace the standard layer
    :param layer_norm_eps: Epsilon value for layer normalization
    """
    super().__init__()

    if custom_encoder is not None:
      self.encoder = custom_encoder
    else:
      encoder_layer = TransformerEncoderLayer(dim_model, num_heads, dim_ff, drop, activation, layer_norm_eps)
      self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, layer_norm, layer_norm_eps)

    if custom_decoder is not None:
      self.decoder = custom_decoder
    else:
      decoder_layer = TransformerDecoderLayer(dim_model, num_heads, dim_ff, drop, activation, layer_norm_eps)
      self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, layer_norm, layer_norm_eps)

    self.norm_eps = layer_norm_eps
    self.dim_model = dim_model
    self.num_heads = num_heads

  def forward(self, src: LayerRef, tgt: LayerRef) -> LayerRef:
    """
    Forward step of Transformer
    """
    memory = self.encoder(src)
    output = self.decoder(tgt, memory)
    return output
