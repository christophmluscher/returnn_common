"""
Base interface for any kind of encoder.

This is basically any generic function x -> y.

Note that in practice, when designing some model,
this interface is even not needed,
because you only care about the final encoded vectors,
and not how you got there.
Automatic differentiation will automatically
also train the encoder.
So, for most purpose, e.g. for a decoder (see :mod:`..decoder.base`),
you only care about some encoded vector of type :class:`LayerRef`.
"""

from ... import nn


class IEncoder(nn.Module):
  """
  Generic encoder interface
  """

  @nn.scoped
  def __call__(self, source: nn.LayerRef) -> nn.LayerRef:
    """
    Encode the input
    """
    raise NotImplementedError
