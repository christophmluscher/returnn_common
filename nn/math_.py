"""
Some basic math functions
(potential activation functions).
"""

from .. import nn


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


def exp(x: nn.LayerRef) -> nn.Layer:
  """exp"""
  return _activation(x, activation="exp")


def log(x: nn.LayerRef) -> nn.Layer:
  """log"""
  return _activation(x, activation="log")


def tanh(x: nn.LayerRef) -> nn.Layer:
  """tanh"""
  return _activation(x, activation="tanh")


def sigmoid(x: nn.LayerRef) -> nn.Layer:
  """sigmoid"""
  return _activation(x, activation="sigmoid")


def log_sigmoid(x: nn.LayerRef) -> nn.Layer:
  """log sigmoid"""
  return _activation(x, activation="log_sigmoid")


def swish(x: nn.LayerRef) -> nn.Layer:
  """swish"""
  return _activation(x, activation="swish")


# softmax already provided via generated layers
softmax = nn.softmax


def log_softmax(x: nn.LayerRef, **kwargs) -> nn.Layer:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, log_space=True, **kwargs)


def _activation(x: nn.LayerRef, activation: str) -> nn.Layer:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  return nn.make_layer({"class": "activation", "from": x, "activation": activation}, name=activation)
