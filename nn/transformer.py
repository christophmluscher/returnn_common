"""
Transformer Modules
"""

from __future__ import annotations
from typing import Optional, Any, Union, Callable
from .. import nn


class TransformerEncoderLayer(nn.Module):
  """
  Defines one layer of a standard transformer encoder
  """
  def __init__(self, dim_model: int, num_heads: int, dim_ff: int = 2048, dropout: float = 0.1,
               activation: Callable[[nn.LayerRef], nn.LayerRef] = nn.relu, norm_eps: float = 1e-5,
               norm_first: bool = False, norm: Callable[[nn.LayerRef], nn.LayerRef] = nn.layer_norm) -> None:
    """
    :param dim_model: hidden dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = MultiheadAttention(dim_model, num_heads, dropout=dropout)

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(dim_model)
    self.activation = activation
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.norm = norm
    self.dropout = dropout

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    """
    Two possible forward variants of encoder, defined by self.norm_first, inp has shape (B, T, F)
    """
    x = inp
    if self.norm_first:
      x = x + self._self_attention_block(self.norm(x, epsilon=self.norm_eps))
      x = x + self._feed_forward_block(self.norm(x, epsilon=self.norm_eps))
    else:
      x = self.norm(x + self._self_attention_block(x), epsilon=self.norm_eps)
      x = self.norm(x + self._feed_forward_block(x), epsilon=self.norm_eps)

    return x

  def _self_attention_block(self, x: nn.LayerRef) -> nn.LayerRef:
    x = self.self_attn(x, x, x)
    return nn.dropout(x, self.dropout)

  def _feed_forward_block(self, x: nn.LayerRef) -> nn.LayerRef:
    x = self.linear_ff(x)
    x = self.activation(x)
    x = nn.dropout(x, dropout=self.dropout)
    x = self.linear_out(x)
    x = nn.dropout(x, dropout=self.dropout)
    return x


class TransformerEncoder(nn.Module):
  """
  Defines the full Encoder of the standard transformer
  """
  def __init__(self, encoder_layer: Union[TransformerEncoderLayer, Any], num_layers: int,
               norm: Callable[[nn.LayerRef], nn.LayerRef] = nn.layer_norm, norm_eps: float = 1e-5):
    """
    :param encoder_layer: Encoder layer to be stacked num_layers times
    :param num_layers: Number of layers
    :param norm: normalization function
    :param norm_eps: Epsilon value for layer normalization
    """
    super().__init__()
    import copy
    self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    self.num_layers = num_layers
    self.norm = norm
    self.norm_eps = norm_eps

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    """
    Applies every encoder layer initialized in self.layers.
    """
    output = inp
    for mod in self.layers:
      output = mod(output)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class TransformerDecoderLayer(nn.Module):
  """
  Defines one layer of a standard transformer decoder
  """
  def __init__(self, dim_model: int, num_heads: int, dim_ff: int = 2048, dropout: float = 0.1,
               activation: Callable[[nn.LayerRef], nn.LayerRef] = nn.relu, norm_eps: float = 1e-5,
               norm_first: bool = False, norm: Callable[[nn.LayerRef], nn.LayerRef] = nn.layer_norm):
    """
    :param dim_model: hidden dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = MultiheadAttention(dim_model, num_heads, dropout=dropout)
    self.multihead_attn = MultiheadAttention(dim_model, num_heads, dropout=dropout)

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(dim_model)

    self.norm = norm
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.activation = activation
    self.dropout = dropout

  def forward(self, tgt: nn.LayerRef, memory: nn.LayerRef) -> nn.LayerRef:
    """
    Two possible forward variants of decoder, defined by self.norm_first, tgt and memory have shape (B, T, F)
    """
    x = tgt
    if self.norm_first:
      x = x + self._self_attention_block(self.norm(x, epsilon=self.norm_eps))
      x = x + self._multi_head_attention_block(self.norm(x, epsilon=self.norm_eps), memory)
      x = x + self._feed_forward_block(self.norm(x, epsilon=self.norm_eps))
    else:
      x = self.norm(x + self._self_attention_block(x), epsilon=self.norm_eps)
      x = self.norm(x + self._multi_head_attention_block(x, memory), epsilon=self.norm_eps)
      x = self.norm(x + self._feed_forward_block(x), epsilon=self.norm_eps)

    return x

  def _self_attention_block(self, x: nn.LayerRef) -> nn.LayerRef:
    x = self.self_attn(x, x, x)
    return nn.dropout(x, self.dropout)

  def _multi_head_attention_block(self, x: nn.LayerRef, mem: nn.LayerRef) -> nn.LayerRef:
    x = self.multihead_attn(x, mem, mem)
    return nn.dropout(x, self.dropout)

  def _feed_forward_block(self, x: nn.LayerRef) -> nn.LayerRef:
    x = self.linear_ff(x)
    x = self.activation(x)
    x = nn.dropout(x, dropout=self.dropout)
    x = self.linear_out(x)
    x = nn.dropout(x, dropout=self.dropout)
    return x


class TransformerDecoder(nn.Module):
  """
  Defines the full Decoder of the standard transformer
  """
  def __init__(self, decoder_layer: Union[TransformerDecoderLayer, Any], num_layers: int,
               norm: Callable = nn.layer_norm, norm_eps: float = 1e-5):
      """
      :param decoder_layer: Decoder layer to be stacked num_layers times
      :param num_layers: Number of layers
      :param norm: normalization function for output layer normalization
      :param norm_eps: Epsilon value for output layer normalization
      """
      super(TransformerDecoder, self).__init__()
      import copy
      self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

      self.num_layers = num_layers
      self.norm = norm
      self.norm_eps = norm_eps

  def forward(self, tgt: nn.LayerRef, memory: nn.LayerRef) -> nn.LayerRef:
    """
    Applies every decoder layer initialized in self.layers.
    """
    output = tgt
    for mod in self.layers:
      output = mod(output, memory)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps)

    return output


class Transformer(nn.Module):
  """
  Standard Transformer Module
  """
  def __init__(self, output_dim: int = 512, num_heads: int = 8, num_encoder_layers: int = 6,
               num_decoder_layers: int = 6, dim_ff: int = 2048, dropout: float = 0.1,
               activation: Callable[[nn.LayerRef], nn.LayerRef] = nn.relu,
               custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
               norm_eps: float = 1e-5, norm: Callable[[nn.LayerRef], nn.LayerRef] = nn.layer_norm) -> None:
    """
    :param output_dim: output dim, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param num_encoder_layers: Number of encoder layers
    :param num_decoder_layers: Number of decoder layers
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param custom_encoder: Custom Encoder layer to replace the standard layer
    :param custom_decoder: Custom Decoder layer to replace the standard layer
    :param norm_eps: Epsilon value for layer normalization
    :param norm: function for layer normalization
    """
    super().__init__()

    if custom_encoder is not None:
      self.encoder = custom_encoder
    else:
      encoder_layer = TransformerEncoderLayer(
        dim_model=output_dim, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout, activation=activation,
        norm_eps=norm_eps, norm=norm)
      self.encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=norm, norm_eps=norm_eps)

    if custom_decoder is not None:
      self.decoder = custom_decoder
    else:
      decoder_layer = TransformerDecoderLayer(
        dim_model=output_dim, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout, activation=activation,
        norm_eps=norm_eps, norm=norm)
      self.decoder = TransformerDecoder(
        decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=norm, norm_eps=norm_eps)

    self.norm_eps = norm_eps
    self.output_dim = output_dim
    self.num_heads = num_heads
    self.norm = norm

  def forward(self, src: nn.LayerRef, tgt: nn.LayerRef) -> nn.LayerRef:
    """
    Forward step of Transformer
    """
    memory = self.encoder(src)
    output = self.decoder(tgt, memory)
    return output
