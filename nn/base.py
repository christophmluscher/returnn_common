"""
Base interfaces.

The core interfaces for the user are:

* :class:`Module` and using :func:`make_layer` to directly create a RETURNN layer via dict.
  We recommend using this only for directly wrapping RETURNN layers
  and not for any higher-level logic,
  which should be done as a :class:`Module`.

* :class:`Module`, to write PyTorch-style code, which acts like a subnetwork.
  We recommend using this as the base interface
  for any higher-level interfaces
  (such as a generic decoder interface).
  Use :func:`scoped` as a decorator for the ``__call__`` method.

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.

The root network should be a :class:`Module`,
and then you can use ``make_root_net_dict()``
to get the network dict.
Code example::

    class Network(nn.Module):
      def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 1024))

      @nn.scoped
      def __call__(self, x: nn.LayerRef) -> nn.Layer:
        y = self.lstm(x)
        return y

    net = Network()
    net_dict = make_root_net_dict(net, "data")

---

Code conventions:

- Usual, as in RETURNN, PEP8, 2-space indents, 120 char line limit.
- Pure interface classes are prefixed with `I`.
  (`Module` is an exception because this is made analogue to PyTorch).

"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Sequence
from returnn.tf.util.data import *  # Dim, Data, and others
# noinspection PyProtectedMember
from returnn.tf.util.data import _MarkedDim
from tensorflow.python.util import nest
from .. import nn


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]
RawTensorTypes = Union[int, float, complex, bool, str]
OutShapeType = Union[Set[Union[Dim, _MarkedDim]], tuple, list]

min_returnn_behavior_version = 12


class LayerRef:
  """
  Refers to a layer.

  An instance of this class can be treated very much like a tensor.
  It supports all the common unary and binary math operations such as addition.
  This is the intended view point for the user,
  to treat instances of this class like a tensor.

  For most layers, instead of just having an instance of :class:`LayerRef`,
  you would instead directly have an instance of :class:`Layer`.

  You do not create instances of this object explicitly
  but they are created via :func:`get_special_layer` or :class:`NameCtx.get_child_layer_ref`,
  or layers (:class:`Layer`) via :func:`make_layer`.
  """

  def __init__(self, *, name_ctx: nn.NameCtx, data: Data):
    self.parent_modules = []  # type: List[Tuple[nn.Module, str]]  # with attr
    self.name_ctx = name_ctx
    assert name_ctx.layer_ref is None
    name_ctx.layer_ref = self
    self.data = data

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx}>"

  @property
  def shape(self) -> Set[Dim]:
    """
    :return: shape (set of dims)
    """
    return self.data.dim_tags_set_implicit

  @property
  def dtype(self) -> str:
    """
    :return: data type (e.g. "float32")
    """
    return self.data.dtype

  @property
  def feature_dim(self) -> Optional[Dim]:
    """
    :return: feature dim
    """
    dim = self.data.feature_dim_or_sparse_dim
    if dim and dim.kind == Dim.Types.Feature:
      # Make sure it is unique.
      feature_dims = [dim_ for dim_ in self.data.dim_tags_set_implicit if dim_.kind == Dim.Types.Feature]
      if feature_dims == [dim]:
        return dim
    return None

  def verify_out_shape(self, out_shape: OutShapeType):
    """
    Verify out_shape via :func:`Data.verify_out_shape`.

    This does not add out_shape to the layer dict as we already have that automatically.
    Thus, this is purely for verification here on returnn-common side.

    :return: self, such that you can write this as a chained op
    :rtype: LayerRef
    """
    self.data.verify_out_shape(out_shape)
    return self

  def get_name_in_current_ctx(self) -> str:
    """
    :return: RETURNN layer name, valid in the current active name context.
    """
    return self.get_name_in_ctx(ctx=nn.NameCtx.current_ctx())

  def get_name_in_ctx(self, ctx: nn.NameCtx) -> str:
    """
    :return: RETURNN layer name in the given name context.
    """
    if not self.name_ctx.parent and ctx != self.name_ctx:
      # We allow creating name ctx early without having a known parent,
      # such as for Parameter, which might be created outside a name context,
      # or in an unrelated name context.
      # We caught this case here, and now assign some parent.
      assert self.parent_modules  # cannot assign parent without parent modules
      for parent_module, attr in self.parent_modules:
        if getattr(parent_module, attr, None) is not self:
          continue  # might have been reset later...
        if parent_module.calls:
          self.name_ctx.assign_parent(parent_module.calls[0], attr)
          break
      assert self.name_ctx.parent, f"{self.parent_modules}"  # could not find parent
    return self.name_ctx.get_name_in_ctx(ctx=ctx)

  def get_abs_name(self) -> str:
    """
    :return: absolute RETURNN layer name starting from root context.
    """
    return self.name_ctx.get_abs_name()

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    raise TypeError(f"mark_as_loss can only be called on a layer, not a layer-ref {self}.")

  def mark_as_default_output(self) -> LayerRef:
    """
    Mark this as the default output, i.e. create the "output" layer with a reference to this.

    :return: the "output" layer.
    """
    return nn.NameCtx.current_ctx().make_default_output(self)

  def mark_as_output(self):
    """
    Mark this as an output.
    """
    raise TypeError(f"mark_as_output can only be called on a layer, not a layer-ref {self}.")

  def __add__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="add", name="add")

  def __sub__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="sub", name="sub")

  def __mul__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="mul", name="mul")

  def __truediv__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="truediv", name="truediv")

  def __radd__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([nn.convert_to_layer_ref(other), self], kind="add", name="add")

  def __rsub__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([nn.convert_to_layer_ref(other), self], kind="sub", name="sub")

  def __rmul__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([nn.convert_to_layer_ref(other), self], kind="mul", name="mul")

  def __rtruediv__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([nn.convert_to_layer_ref(other), self], kind="truediv", name="truediv")

  def __neg__(self) -> Layer:
    from ._generated_layers import _eval
    return _eval(self, eval="-source(0)", name="neg")

  def __invert__(self) -> Layer:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.logical_not(source(0))", name="invert")

  def __pow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> Layer:
    assert modulo is None
    from ._generated_layers import _eval
    return _eval([self, nn.convert_to_layer_ref(other)], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __rpow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> Layer:
    assert modulo is None
    from ._generated_layers import _eval
    return _eval([nn.convert_to_layer_ref(other), self], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __and__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="logical_and", name="logical_and")

  def __or__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _combine
    return _combine([self, nn.convert_to_layer_ref(other)], kind="logical_or", name="logical_or")

  def __abs__(self) -> Layer:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.abs(source(0))", name="abs")

  def __ceil__(self) -> Layer:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.math.ceil(source(0))", name="ceil")

  def __floor__(self) -> Layer:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.math.floor(source(0))", name="floor")

  def __floordiv__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _eval
    return _eval([self, nn.convert_to_layer_ref(other)], eval="tf.math.floordiv(source(0), source(1))", name="floordiv")

  def __eq__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="equal", name="equal")

  def __ne__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="not_equal", name="not_equal")

  def __lt__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="less", name="less")

  def __le__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="less_equal", name="less_equal")

  def __gt__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="greater", name="greater")

  def __ge__(self, other: Union[RawTensorTypes, LayerRef]) -> Layer:
    from ._generated_layers import _compare
    return _compare([self, nn.convert_to_layer_ref(other)], kind="greater_equal", name="greater_equal")


class Layer(LayerRef):
  """
  Represents a layer and its output, created by :func:`make_layer`.
  You would not create an instance of this explicitly.
  """

  def __init__(self, *, layer_dict: LayerDictRaw, name_ctx: nn.NameCtx,
               predefined_out_data: Optional[Data] = None,
               add_out_shape_info: bool = True):
    if predefined_out_data:
      data = predefined_out_data
    else:
      data = _data_from_layer_dict(layer_dict)
    if add_out_shape_info and layer_dict["class"] not in {"constant", "variable"}:
      layer_dict["out_shape"] = set(data.dim_tags_set_implicit)
    super(Layer, self).__init__(name_ctx=name_ctx, data=data)
    assert self.name_ctx.layer is None
    self.name_ctx.layer = self
    self.layer_dict = layer_dict

  def __repr__(self):
    return (
      f"<{self.__class__.__name__}"
      f" {self.name_ctx.get_abs_name_repr()}"
      f" {self.data.__repr__()}"
      f" via {self.name_ctx.module if self.name_ctx.module else self.layer_dict.get('class', '?')!r}>")

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    assert "loss" not in self.layer_dict
    self.layer_dict["loss"] = "as_is"
    if loss_scale is not None:
      assert "loss_scale" not in self.layer_dict
      self.layer_dict["loss_scale"] = loss_scale

  def mark_as_output(self):
    """
    Mark this as an output.
    """
    scope = nn.NameCtx.current_ctx()
    assert scope.parent is None, f"{self} mark_as_output only makes sense at the top level"
    self.layer_dict["is_output_layer"] = True
    scope.marked_outputs.append(self)

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper  # noqa
    return sis_hash_helper(self.layer_dict)


class Parameter(Layer):
  """
  This represents a (potential trainable) parameter,
  aka ``tf.Variable`` in TensorFlow,
  wrapping to ``VariableLayer`` in RETURNN.
  """
  def __init__(self, shape: Sequence[Dim], dtype: Optional[str] = None,
               *,
               trainable: Optional[bool] = None,
               auxiliary: bool = False):
    """
    :param shape:
    :param dtype:
    :param trainable: if True, and optimizer would do updates to this parameter in training mode
    :param auxiliary: if True, this indicates that this parameter should not be transformed by transformations
      such as weight normalization. One example are running statistics, as used for batch normalization.
      This usually implies that the parameter is not trainable, i.e. not to be updated by the optimizer,
      but usually has some custom update.
      This flag is not passed on to RETURNN but just used here for returnn-common logic.
    """
    if not all(isinstance(dim, Dim) for dim in shape):
      raise TypeError(f"shape {shape} must be a sequence of Dim")
    if not all(isinstance(dim.dimension, int) for dim in shape):
      raise ValueError(f"shape {shape} must be static")
    if len(shape) != len(set(shape)):
      raise ValueError(f"shape {shape} dims must be unique")
    # Note: At creation time, we don't know the name yet.
    # The name will be inferred by the parent modules and the attribute chain.
    name_ctx = nn.NameCtx(name="parameter", parent=None)  # this is incomplete and will be configured later
    data = Data("parameter", dim_tags=list(shape), dtype=dtype)
    layer_dict = {"class": "variable", "shape": list(shape)}
    if dtype is not None:
      layer_dict["dtype"] = dtype
    if auxiliary and trainable is None:
      trainable = False
    if trainable is not None:
      layer_dict["trainable"] = trainable
    super(Parameter, self).__init__(
      layer_dict=layer_dict,
      predefined_out_data=data, add_out_shape_info=False,
      name_ctx=name_ctx)
    self.auxiliary = auxiliary


class LayerState(dict):
  """
  Covers all the state of a layer,
  i.e. exactly what needs to be stored and passed into the module or module
  next time you call it as initial state.

  This behaves somewhat like a namedtuple, although we derive from dict.
  """
  def __init__(self, *args, **kwargs):
    if kwargs:
      assert not args
      super().__init__(**kwargs)
    elif args:
      assert len(args) == 1
      if isinstance(args[0], dict):
        super().__init__(**args[0])
      else:
        super().__init__(state=args[0])
    else:
      super().__init__()

  def __repr__(self):
    return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for (k, v) in self.items())})"

  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"{self}.{item}")

  def __setattr__(self, key, value):
    self[key] = value


def make_layer(layer_dict: LayerDictRaw, *,
               name: Optional[Union[str, nn.NameCtx]] = None,
               module: Optional[nn.Module] = None,
               predefined_out_data: Optional[Data] = None,
               add_out_shape_info: bool = True) -> Layer:
  """
  Creates the layer. This also registers the layer instance in the top name ctx.
  When no name is given, this assumes that the top name ctx corresponds to this module.

  If a layer has params, and you want the param sharing logic,
  you should instead derive a new class from :class:`Module`.
  Usually, you do not need either of these,
  as all standard layers should already be wrapped,
  and it should be possible to define any possible logic
  using that.
  (If this is not the case, please report an issue.)

  :param LayerDictRaw layer_dict: can contain :class:`LayerRef` instances
  :param str|NameCtx|None name:
    if str: (suggested) layer name. if given, will create a new :class:`NameCtx`
    if NameCtx, will use this.
  :param Module|None module: if given, will create new name scope with this module
  :param Data|None predefined_out_data: normally we can derive the out data automatically.
    If this should be skipped, you can pass this explicitly.
  :param bool add_out_shape_info: if True, we will add the out shape info to the layer.
  """
  if isinstance(name, str) or module:
    assert not name or isinstance(name, str)
    name_ctx = nn.NameCtx.get_from_call(module=module, name=name)
    return make_layer(
      layer_dict=layer_dict, name=name_ctx,
      predefined_out_data=predefined_out_data, add_out_shape_info=add_out_shape_info)
  elif isinstance(name, nn.NameCtx):
    name_ctx = name
    if nn.NameCtx.top() is name:
      pass  # go on
    else:
      with name_ctx:
        return make_layer(
          layer_dict=layer_dict, predefined_out_data=predefined_out_data, add_out_shape_info=add_out_shape_info)
  else:
    name_ctx = nn.NameCtx.top()
  assert not name_ctx.layer_ref and not name_ctx.layer  # not yet assigned
  layer_dict = layer_dict.copy()

  if name_ctx.module and name_ctx.module.has_parameters:
    # We must check whether the RETURNN abs layer name is consistent with our module naming hierarchy,
    # and make it consistent if not (https://github.com/rwth-i6/returnn_common/issues/25).
    if name_ctx.is_root:
      pass  # nothing to do
    else:
      # The parent name ctx RETURNN layer will also have the right name_scope set,
      # so this layers name scope default is simply based on that.
      layer_abs_name_scope_parent = name_ctx.parent.layer_abs_name_scope
      if layer_abs_name_scope_parent:
        layer_abs_name_scope_parent += "/"
      layer_abs_name_scope_default = layer_abs_name_scope_parent + name_ctx.name
      if layer_abs_name_scope_default != name_ctx.layer_abs_name_scope:  # default does not match what we require
        assert "name_scope" not in layer_dict
        if name_ctx.layer_abs_name_scope == name_ctx.parent.layer_abs_name_scope:
          layer_dict["name_scope"] = ""
        elif name_ctx.layer_abs_name_scope.startswith(layer_abs_name_scope_parent):  # can use relative
          layer_dict["name_scope"] = name_ctx.layer_abs_name_scope[len(layer_abs_name_scope_parent):]
        else:  # must use absolute
          layer_dict["name_scope"] = "/" + name_ctx.layer_abs_name_scope

  name_ctx.is_subnet_ctx = False
  layer = Layer(
    layer_dict=layer_dict, name_ctx=name_ctx,
    predefined_out_data=predefined_out_data, add_out_shape_info=add_out_shape_info)
  if name_ctx.module:
    name_ctx.module.calls.append(name_ctx)
  return layer


def get_extern_data(data: Data) -> LayerRef:
  """
  Get extern data from root ctx.
  As a side effect, it registers the given data as extern data,
  and this will be included when creating the RETURNN config,
  via :func:`NameCtx.get_returnn_config`.
  """
  assert isinstance(data, Data)  # the usage was different before. make sure we get this correct
  scope = nn.NameCtx.top()  # must exist
  assert not scope.parent  # get_extern_data only allowed (only makes sense) in root name ctx
  if data.name not in scope.extern_data:
    scope.extern_data[data.name] = data
  else:
    assert scope.extern_data[data.name] is data
  root_layer_name = f"data:{data.name}"
  return _get_special_layer(root_layer_name, scope=scope, data=data)


def _get_special_layer(name: str, *, scope: Optional[nn.NameCtx] = None, data: Data) -> LayerRef:
  """
  Special layer can be "data:..." or whatever.
  """
  if not scope:
    scope = nn.NameCtx.current_ctx()  # must exist
  return scope.get_child_layer_ref(name, data=data)


def _get_sub_layer(layer: LayerRef, name: str, *, data: Data) -> LayerRef:
  """
  Like the "{layer}/{name}" syntax in RETURNN.
  Normally this should only be needed for internal usage.
  """
  return layer.name_ctx.get_child_layer_ref(name, data=data)


def _data_from_layer_dict(layer_dict: LayerDictRaw) -> Data:
  """
  Use RETURNN layer_class.get_out_data_from_opts to get the :class:`Data`.
  For this function, we need to set up some dummy network and dummy source layers.
  """
  from returnn.tf.network import TFNetwork, ExternData, get_layer_class
  from returnn.tf.layers.base import InternalLayer, LayerBase
  from returnn.util import BehaviorVersion
  from returnn.config import Config
  config = Config({
    "behavior_version": min_returnn_behavior_version,
  })
  BehaviorVersion.set(min_returnn_behavior_version)
  ctx = nn.NameCtx.top()
  inside_rec_time_dim = None
  while ctx:
    mod = ctx.module
    if isinstance(mod, nn.LoopModule):
      inside_rec_time_dim = mod.loop.axis
      break
    ctx = ctx.parent
  net = TFNetwork(config=config, extern_data=ExternData(), name="dummy_net", inside_rec_time_dim=inside_rec_time_dim)

  ref_to_layer_name = {}  # type: Dict[nn.NameCtx, str]

  def _get_unique_name(name) -> str:
    reserved_names = set(net.layers.keys()) | {"data"}
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1

  def _get_layer_name(ref: LayerRef) -> str:
    if ref.name_ctx in ref_to_layer_name:
      return ref_to_layer_name[ref.name_ctx]
    name = _get_unique_name(ref.name_ctx.name)
    ref_to_layer_name[ref.name_ctx] = name
    assert name not in net.layers
    net.layers[name] = InternalLayer(name=name, network=net, output=ref.data)
    return name

  def _get_layer(name: str) -> LayerBase:
    assert name in net.layers
    return net.layers[name]

  def _map_layer_dict_elem(value):
    if isinstance(value, LayerRef):
      return _get_layer_name(value)
    return value

  layer_dict = nest.map_structure(_map_layer_dict_elem, layer_dict)
  out_name = _get_unique_name("output")

  layer_desc = layer_dict.copy()
  layer_class = get_layer_class(layer_desc.pop("class"))
  # Note about name:
  # The name can be to the root network (full name) or to the owning/direct network (`net`) (base_name).
  # The name can optionally have a prefix (here we only care about extra net prefix "extra...:").
  # The prefix is implied by the owning network.
  layer_desc["_network"] = net
  layer_desc["_name"] = out_name

  layer_class.transform_config_dict(layer_desc, network=net, get_layer=_get_layer)

  # noinspection PyProtectedMember
  layer_desc = net._create_layer_layer_desc(name=out_name, layer_desc=layer_desc, template=True)
  out_data = layer_class.get_out_data_from_opts(**layer_desc)

  return out_data
