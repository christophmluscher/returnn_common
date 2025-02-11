"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union
from .. import nn
from ._generated_layers import _eval


def identity(x: nn.LayerRef) -> nn.LayerRef:
  """
  Identity function. Just to have one canonical.
  Also see :func:`nn.copy`, which creates a new layer (which itself does nothing though).
  """
  return x


def relu(x: nn.LayerRef) -> nn.Layer:
  """ReLU"""
  return _activation(x, activation="relu")


def elu(x: nn.LayerRef) -> nn.Layer:
  """ELU https://arxiv.org/abs/1511.07289"""
  return _activation(x, activation="elu")


def selu(x: nn.LayerRef) -> nn.Layer:
  """SELU https://arxiv.org/abs/1706.02515"""
  return _activation(x, activation="selu")


def gelu(x: nn.LayerRef) -> nn.Layer:
  """GELU https://arxiv.org/abs/1606.08415"""
  return _activation(x, activation="gelu")


@nn.scoped
def glu(x: nn.LayerRef, axis: nn.Dim) -> nn.Layer:
  """GLU https://arxiv.org/abs/1612.08083"""
  from . import split
  a, b = split(x, axis=axis, out_dims=[axis // 2, axis // 2])
  return a * sigmoid(b)


def exp(x: nn.LayerRef) -> nn.Layer:
  """exp. see also :func:`safe_exp`"""
  return _activation(x, activation="exp")


def safe_exp(x: nn.LayerRef, *, eps: float = 1e-7) -> nn.Layer:
  """
  exp (:func:`exp`) with extra logic
    replacing earlier log_softmax by softmax, log_sigmoid by sigmoid, log by identity, etc.
  Also, for the fallback exp, clips the min and max value.

  Note that the default eps is higher than the default in RETURNN.
  """
  return _eval(x, eval=f"safe_exp(source(0), eps={eps!r})", name="safe_exp")


def log(x: nn.LayerRef) -> nn.Layer:
  """log. see also :func:`safe_log`"""
  return _activation(x, activation="log")


def safe_log(x: nn.LayerRef, *, eps: float = 1e-7, use_fake_grad: bool = True) -> nn.Layer:
  """
  log (:func:`log`) with extra logic
    replacing earlier softmax by log_softmax, sigmoid by log_sigmoid, exp by identity, etc.
  Also, for the fallback log, adds some eps in the backprop (only in backprop) to avoid nan/inf.

  Note that the default eps is higher than the default in RETURNN.
  """
  return _eval(x, eval=f"safe_log(source(0), eps={eps!r}, use_fake_grad={use_fake_grad!r})", name="safe_log")


def tanh(x: nn.LayerRef) -> nn.Layer:
  """tanh"""
  return _activation(x, activation="tanh")


def sigmoid(x: nn.LayerRef) -> nn.Layer:
  """sigmoid"""
  return _activation(x, activation="sigmoid")


def log_sigmoid(x: nn.LayerRef) -> nn.Layer:
  """log sigmoid"""
  return _activation(x, activation="log_sigmoid")


def sqrt(x: nn.LayerRef) -> nn.Layer:
  """sqrt"""
  return _activation(x, activation="sqrt")


def rsqrt(x: nn.LayerRef) -> nn.Layer:
  """rsqrt"""
  return _activation(x, activation="rsqrt")


def swish(x: nn.LayerRef) -> nn.Layer:
  """swish"""
  return _activation(x, activation="swish")


def squared_difference(a: nn.LayerRef, b: nn.LayerRef, *, name: Optional[str] = None) -> nn.Layer:
  """wraps tf.math.squared_difference"""
  return _eval([a, b], eval="tf.math.squared_difference(source(0), source(1))", name=name or "squared_difference")


# softmax already provided via generated layers


def log_softmax(x: nn.LayerRef, *, axis: nn.Dim, **kwargs) -> nn.Layer:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, axis=axis, log_space=True, **kwargs)


def _activation(x: nn.LayerRef, activation: str) -> nn.Layer:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  return nn.make_layer({"class": "activation", "from": x, "activation": activation}, name=activation)


def cumsum(
      x: nn.LayerRef, *,
      axis: nn.Dim,
      additional_left_summand_per_element: Optional[Union[str, int, float]] = nn.NotSpecified,
      reverse: bool = nn.NotSpecified,
      name: Optional[str] = None) -> nn.Layer:
  """
  Applies cumsum.
  See :func:`._generated_layers._cumsum`.
  """
  from ._generated_layers import rec_cum_sum
  layer, state = rec_cum_sum(
    x, axis=axis,
    additional_left_summand_per_element=additional_left_summand_per_element,
    reverse=reverse,
    name=name)
  del state
  return layer
