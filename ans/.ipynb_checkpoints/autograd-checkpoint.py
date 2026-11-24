from collections.abc import Callable
from numbers import Number
from typing import Self

import torch


class Variable:
  def __init__(
      self,
      data: torch.Tensor,
      parents: tuple[Self, ...] = (),
      grad_fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]] | None = None,
      name: str | None = None,
  ) -> None:
    if not isinstance(data, torch.Tensor):
      data = torch.tensor(data)
    self.data = data
    self.grad: torch.Tensor | None = None
    self.parents = parents
    self.grad_fn = grad_fn
    self.name = name

  def __repr__(self):
    if hasattr(self.grad_fn, "func"):
      grad_fn_repr = self.grad_fn.func.__qualname__
    elif self.grad_fn is not None:
      grad_fn_repr = self.grad_fn.__qualname__
    else:
      grad_fn_repr = "None"
    if self.data.ndim == 0 or (
        self.data.shape[-1] == self.data.numel() and self.data.numel() < 5
    ):
      return f"{type(self).__name__}({self.data}, grad_fn={grad_fn_repr})"
    else:
      return f"{type(self).__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})"

  def _arg_to_variable(self, arg: Number | torch.Tensor | Self) -> Self:
    if isinstance(arg, Variable):
      return arg
    else:
      if not isinstance(arg, torch.Tensor):
        arg = torch.tensor(arg, dtype=self.data.dtype, device=self.data.device)
      return type(self)(arg)

  def _unbroadcast(
      self, grad: torch.Tensor, target_shape: torch.Size
  ) -> torch.Tensor:
    if grad.shape == target_shape:
      return grad
    sum_dims = []
    grad_ndim = grad.ndim
    target_ndim = len(target_shape)
    for i in range(grad_ndim - target_ndim):
      sum_dims.append(i)
    for i, size in enumerate(target_shape):
      if size == 1:
        sum_dims.append(i + (grad_ndim - target_ndim))
    if sum_dims:
      grad = grad.sum(dim=sum_dims, keepdim=True)
    if grad.shape != target_shape:
      grad = grad.reshape(target_shape)
    return grad

  def __add__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)

    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      grad_self = self._unbroadcast(dout, self.data.shape)
      grad_other = self._unbroadcast(dout, other_var.data.shape)
      return grad_self, grad_other

    return type(self)(
        self.data + other_var.data,
        parents=(self, other_var),
        grad_fn=grad_fn,
    )
    # ENDTODO
    ########################################

  def __radd__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)
    return other_var + self
    # ENDTODO
    ########################################

  def __sub__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)

    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      grad_self = self._unbroadcast(dout, self.data.shape)
      grad_other = self._unbroadcast(-dout, other_var.data.shape)
      return grad_self, grad_other

    return type(self)(
        self.data - other_var.data,
        parents=(self, other_var),
        grad_fn=grad_fn,
    )
    # ENDTODO
    ########################################

  def __rsub__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)
    return other_var - self
    # ENDTODO
    ########################################

  def __mul__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)

    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      grad_self = self._unbroadcast(dout * other_var.data, self.data.shape)
      grad_other = self._unbroadcast(dout * self.data, other_var.data.shape)
      return grad_self, grad_other

    return type(self)(
        self.data * other_var.data,
        parents=(self, other_var),
        grad_fn=grad_fn,
    )
    # ENDTODO
    ########################################

  def __rmul__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)
    return other_var * self
    # ENDTODO
    ########################################

  def __truediv__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)

    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      grad_self = self._unbroadcast(dout / other_var.data, self.data.shape)
      grad_other = self._unbroadcast(
          -dout * self.data / (other_var.data**2), other_var.data.shape
      )
      return grad_self, grad_other

    return type(self)(
        self.data / other_var.data,
        parents=(self, other_var),
        grad_fn=grad_fn,
    )
    # ENDTODO
    ########################################

  def __rtruediv__(self, other: Number | torch.Tensor | Self) -> Self:
    ########################################
    other_var = self._arg_to_variable(other)
    return other_var / self
    # ENDTODO
    ########################################

  def __matmul__(self, other: Number | torch.Tensor | Self) -> Self:
        
        other_var = self._arg_to_variable(other)
        out_data = self.data @ other_var.data

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.data.detach().requires_grad_(True)
            y = other_var.data.detach().requires_grad_(True)
            z = x @ y
            grad_x, grad_y = torch.autograd.grad(z, (x, y), dout, allow_unused=True)
            if grad_x is None:
                grad_x = torch.zeros_like(self.data)
            if grad_y is None:
                grad_y = torch.zeros_like(other_var.data)
            return (
                self._unbroadcast(grad_x, self.data.shape),
                self._unbroadcast(grad_y, other_var.data.shape),
            )

        return type(self)(
            out_data,
            parents=(self, other_var),
            grad_fn=grad_fn,
        )

  def __rmatmul__(self, other: Number | torch.Tensor | Self) -> Self:
        ########################################
        other_var = self._arg_to_variable(other)
        return other_var @ self
        # ENDTODO
        ########################################

  def __getitem__(self, item) -> Self:
    selection = self.data[item]
    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
            grad_self = torch.zeros_like(self.data)
            grad_self[item] = dout
            return (grad_self,)
    return type(self)(
            selection,
            parents=(self,),
            grad_fn=grad_fn,
    )

    # ENDTODO
    ########################################

  def log(self) -> Self:
    tensorLog = torch.log(self.data)
    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
        return (dout * (1/self.data),)
    return type(self)(
            tensorLog,
            parents=(self,),
            grad_fn=grad_fn,
    )

    # ENDTODO
    ########################################

  def exp(self) -> Self:
    expp = torch.exp(self.data)
    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
        return (dout * expp,)
    return type(self)(
            expp,
            parents=(self,),
            grad_fn=grad_fn,
    )

    # ENDTODO
    ########################################

  def sigmoid(self) -> Self:
      sigm = 1/(1+torch.exp(-self.data))
      def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
          return (dout * sigm * (1.0-sigm),)
      return type(self)(
            sigm,
            parents=(self,),
            grad_fn=grad_fn,
        )
    
    # ENDTODO
    ########################################

  def relu(self, negative_slope: float = 0.0) -> Self:
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################

  def sum(
      self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
  ) -> Self:
    out_data = (
        self.data.sum()
        if dim is None
        else self.data.sum(dim=dim, keepdim=keepdim)
    )

    def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor]:
      grad = dout
      if dim is None:
        grad = grad.expand(self.data.shape)
      else:
        if not keepdim:
          dims = (dim,) if isinstance(dim, int) else dim
          for d in sorted(d % self.data.ndim for d in dims):
            grad = grad.unsqueeze(d)
      grad = grad.expand(self.data.shape)
      return (grad,)

    return type(self)(
        out_data,
        parents=(self,),
        grad_fn=grad_fn,
    )

    # ENDTODO
    ########################################

  def mean(
      self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False
  ) -> Self:
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################

  def reshape(self, *shape: int) -> Self:
    ########################################
    # TODO: implement

    raise NotImplementedError

    # ENDTODO
    ########################################

  def backprop(self, dout: torch.Tensor | None = None) -> None:
    """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype
        """

    ########################################
    if dout is None:
      dout = torch.ones_like(self.data)
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for parent in v.parents:
          build_topo(parent)
        topo.append(v)

    build_topo(self)
    self.grad = dout
    for v in reversed(topo):
      if v.grad_fn is not None and v.grad is not None:
        grads = v.grad_fn(v.grad)
        for parent, grad in zip(v.parents, grads):
          if parent.grad is None:
            parent.grad = grad
          else:
            parent.grad = parent.grad + grad
    # ENDTODO
    ########################################