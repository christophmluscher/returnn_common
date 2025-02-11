"""
Array (Tensor) functions
"""

from typing import Optional, Tuple, List, Union
from returnn.util.basic import NotSpecified
from .. import nn


def expand_dim(source: nn.LayerRef, *, dim: nn.Dim, name: Optional[str] = None) -> nn.Layer:
  """
  Expand the source by the given dimension,
  which should be 1.

  Note that this is *never* needed for broadcasting.
  All broadcasting should always happen automatically.

  This might be needed for convolution.

  This can be reversed via :func:`squeeze`.
  """
  if dim.dimension != 1:
    raise ValueError(f"{dim} is not a 1-dim")
  # We use SplitDimsLayer for this.
  # ExpandDimsLayer in RETURNN currently would allow to use a dim tag.
  # Now search for a good axis to split via some heuristics.
  source_dims = [d for d in source.data.dim_tags if not d.is_batch_dim()]
  if not source_dims:
    # Unfortunately, for scalars (ignoring batch), split_dims does not work.
    return source + nn.zeros(source.data.dim_tags + (dim,), dtype=source.dtype)
  if dim.is_spatial_dim():
    if any(d.is_spatial_dim() for d in source_dims):
      axis = [d for d in source_dims if d.is_spatial_dim()][-1]
      return nn.split_dims(source, axis=axis, dims=(axis, dim), name=name)
    else:
      axis = source_dims[0]
      return nn.split_dims(source, axis=axis, dims=(dim, axis), name=name)
  elif dim.is_feature_dim():
    if any(d.is_feature_dim() for d in source_dims):
      axis = [d for d in source_dims if d.is_feature_dim()][-1]
      return nn.split_dims(
        source, axis=axis, dims=(axis.copy(same_as_self=True, kind=nn.Dim.Types.Spatial), dim), name=name)
    else:
      axis = source_dims[-1]
      return nn.split_dims(source, axis=axis, dims=(axis, dim), name=name)
  else:
    raise ValueError(f"{dim} is not a spatial or feature dim")


def concat(*sources: Tuple[nn.LayerRef, nn.Dim],
           allow_broadcast=False,
           name: Optional[str] = None) -> nn.Layer:
  """
  Concatenates multiple sources (by default in feature axis).
  """
  opts = {}
  if allow_broadcast:
    opts["allow_broadcast"] = True
  return nn.make_layer({"class": "concat", "from": sources, **opts}, name=name or "concat")


def cum_concat_step(
      source: nn.LayerRef, *, state: nn.LayerState,
      out_spatial_dim: Optional[nn.Dim] = None,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.Dim, nn.LayerState]:
  """
  Concatenates all previous frames of a time-axis.
  See RETURNN :class:`CumConcatLayer` for details.
  """
  from ._generated_layers import rec_cum_concat
  return rec_cum_concat(
    source=source, axis=nn.single_step_dim,
    state=state, out_spatial_dim=out_spatial_dim, name=name)


def split(source: nn.LayerRef, *,
          axis: nn.Dim,
          out_dims: Union[List[nn.Dim], Tuple[nn.Dim, ...]],
          name: Optional[str] = None) -> Tuple[nn.LayerRef, ...]:
  """
  Split the input on the specified axis (by default feature).
  Basically a wrapper around tf.split.
  """
  from ._generated_layers import _split
  from .base import _get_sub_layer
  res = _split(source, axis=axis, out_dims=out_dims, name=name)
  src_axis_int = source.data.get_axis_from_description(axis)
  return tuple(
    _get_sub_layer(
      layer=res, name=str(i),
      data=source.data.copy_template_replace_dim_tag(
        axis=src_axis_int, new_dim_tag=dim,
        name=f"{source.data.name}/split:{i}:{dim.description}"))
    for i, dim in enumerate(out_dims))


def window(
      source: nn.LayerRef, *,
      axis: nn.Dim,
      window_dim: nn.Dim,
      window_left: Optional[int] = NotSpecified,
      window_right: Optional[int] = NotSpecified,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.Dim]:
  """
  Window. See :func:`_generated_layers._window`.
  """
  from ._generated_layers import rec_window
  layer, (window_dim, out_spatial_dim), state = rec_window(
    source,
    window_dim=window_dim, window_left=window_left, window_right=window_right,
    axis=axis, padding=padding, stride=stride,
    name=name)
  del state
  return layer, out_spatial_dim


def window_step(
      source: nn.LayerRef, *, state: nn.LayerState,
      window_dim: nn.Dim,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.LayerState]:
  """
  Window into the past when iterating.
  See :func:`_generated_layers._window`.
  """
  from ._generated_layers import rec_window
  out, _, state = rec_window(
    source, state=state,
    window_dim=window_dim, window_left=window_dim.dimension - 1, window_right=0,
    axis=nn.single_step_dim,
    name=name)
  return out, state
