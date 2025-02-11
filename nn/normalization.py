"""
Normalization functions such as batch norm
"""


from typing import Optional, Sequence, Union, Tuple
from .. import nn


@nn.scoped
def moments(x: nn.LayerRef, axis: Union[nn.Dim, Sequence[nn.Dim]]) -> Tuple[nn.Layer, nn.Layer]:
  """
  :param x: input
  :param axis: the axis to be reduced, to calculate statistics over
  :return: mean, variance. it has the same shape as the input with the axis removed
  """
  mean = nn.reduce(x, mode="mean", axis=axis, name="mean")
  # stop_gradient does not change the gradient here
  variance = nn.reduce(
    nn.squared_difference(x, nn.stop_gradient(mean)),
    mode="mean", axis=axis, name="variance")
  return mean, variance


class BatchNorm(nn.Module):
  """
  Batch normalization. https://arxiv.org/abs/1502.03167

  Note that the default arguments differ from corresponding batch norm in RETURNN.
  See here for discussion on defaults: https://github.com/rwth-i6/returnn/issues/522

  We calculate statistics over all axes except the given in_dim.
  I.e. all other axes are reduced for the statistics.

  To compensate the normalization, there are learnable parameters gamma and beta
  (optional, used when option `affine` is True).

  The usual behavior depends on whether this is used in training or evaluation,
  although this often configurable in other frameworks.
  The usual behavior, in training::

      # Using statistics from current batch.
      mean_cur_batch, variance_cur_batch = moments(source, reduce_dims)
      y = (x - mean_cur_batch) / sqrt(variance_cur_batch + epsilon)
      y = gamma * y + beta

      # Updating running statistics for later use.
      mean = (1 - momentum) * mean + momentum * mean_cur_batch
      variance = (1 - momentum) * variance + momentum * variance_cur_batch

  The usual behavior, not in training (i.e. in evaluation)::

      # Using collected statistics. Not using statistics from current batch.
      y = (x - mean) / sqrt(variance + epsilon)
      y = gamma * y + beta

  """

  def __init__(self, in_dim: Optional[nn.Dim] = None, *,
               affine: bool = True,
               momentum: float = 0.1, epsilon: float = 1e-3,
               use_mask: Optional[bool] = None,
               ):
    """
    :param in_dim: the feature dimension of the input
    :param affine: whether to use learnable parameters gamma and beta
    :param momentum: momentum for the running mean and variance
    :param epsilon: epsilon for the variance
    :param use_mask: whether to use a mask for dynamic spatial dims.
      This must be specified if the input has dynamic spatial dims.
      True would use the correct masking then. However, that is inconsistent to all other frameworks
        which ignore the masking, and also slower, and the fused op would not be used.
      False would be consistent to all other frameworks,
        and potentially allows for the use of an efficient fused op internally.
    """
    super().__init__()
    self.in_dim = in_dim
    self.running_mean = None  # type: Optional[nn.Parameter]
    self.running_variance = None  # type: Optional[nn.Parameter]
    self.affine = affine
    self.gamma = None  # type: Optional[nn.Parameter]
    self.beta = None  # type: Optional[nn.Parameter]
    self.use_mask = use_mask
    self.momentum = momentum
    self.epsilon = epsilon
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    self.in_dim = in_dim
    self.running_mean = nn.Parameter([in_dim], auxiliary=True)
    self.running_variance = nn.Parameter([in_dim], auxiliary=True)
    if self.affine:
      self.gamma = nn.Parameter([in_dim])
      self.beta = nn.Parameter([in_dim])

  @nn.scoped
  def __call__(self, source: nn.LayerRef) -> nn.Layer:
    source = nn.check_in_feature_dim_lazy_init(source, self.in_dim, self._lazy_init)
    # We wrap the RETURNN layer because we want efficient handling if possible,
    # which is potentially the use of a fused op,
    # and maybe reordering of dims.
    # https://github.com/rwth-i6/returnn_common/issues/89
    spatial_dims = source.shape - {nn.batch_dim, self.in_dim}
    assert len(spatial_dims) == len(source.shape) - 2
    if any(d.dimension is None for d in spatial_dims):  # any dynamic spatial dim
      if self.use_mask is None:
        raise ValueError(
          f"{self}: use_mask must be specified if the input {source} has any dynamic spatial dims")
      use_mask = self.use_mask
    else:
      use_mask = False  # not needed. False because this potentially enables an efficient fused op.
    reuse_params = {
      "batch_norm/v2_mean": self.running_mean,
      "batch_norm/v2_variance": self.running_variance,
    }
    if self.affine:
      reuse_params["batch_norm/v2_gamma"] = self.gamma
      reuse_params["batch_norm/v2_beta"] = self.beta
    reuse_params = {"map": {k: {"layer_output": v} for k, v in reuse_params.items()}}
    return nn.make_layer({
      "class": "batch_norm", "from": source, "in_dim": self.in_dim,
      "use_std": self.affine, "use_shift": self.affine,
      "param_version": 2, "reuse_params": reuse_params,
      "momentum": self.momentum, "epsilon": self.epsilon, "masked_time": use_mask,
    }, name="batch_norm")
