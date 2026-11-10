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
        if self.data.ndim == 0 or (self.data.shape[-1] == self.data.numel() and self.data.numel() < 5):
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

    def _unbroadcast(self, grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
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

    def __radd__(self, other: Number | torch.Tensor | Self) -> Self:
        other_var = self._arg_to_variable(other)
        return other_var + self

    def __sub__(self, other: Number | torch.Tensor | Self) -> Self:
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

    def __rsub__(self, other: Number | torch.Tensor | Self) -> Self:
        other_var = self._arg_to_variable(other)
        return other_var - self

    def __mul__(self, other: Number | torch.Tensor | Self) -> Self:
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

    def __rmul__(self, other: Number | torch.Tensor | Self) -> Self:
        other_var = self._arg_to_variable(other)
        return other_var * self

    def __truediv__(self, other: Number | torch.Tensor | Self) -> Self:
        other_var = self._arg_to_variable(other)
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            grad_self = self._unbroadcast(dout / other_var.data, self.data.shape)
            grad_other = self._unbroadcast(-dout * self.data / (other_var.data ** 2), other_var.data.shape)
            return grad_self, grad_other
        return type(self)(
            self.data / other_var.data,
            parents=(self, other_var),
            grad_fn=grad_fn,
        )

    def __rtruediv__(self, other: Number | torch.Tensor | Self) -> Self:
        other_var = self._arg_to_variable(other)
        return other_var / self

    def backprop(self, dout: torch.Tensor | None = None) -> None:
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