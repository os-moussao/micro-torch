from __future__ import annotations
from typing import overload
import numpy as np


class Grad:
    def __init__(self, val, required_grad=True, children=()):
        self.val: np.float64 = val
        self.grad: np.float64 | None = 0 if required_grad else None
        self.requires_grad = required_grad

        self._children = children
        self._backward = None

    def backprop(self):
        """This function computed the partial derivatives of the result to every involved input (which requires gradient)"""

        assert self.requires_grad, "Must require gradient"

        topo: list[Grad] = []
        vis = set()

        def build_topo(v: Grad):
            vis.add(id(v))
            for c in v._children:
                if id(c) not in vis:
                    build_topo(c)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            if node._backward is not None:
                node._backward()

    def __add__(self, other) -> Grad:
        other = other if isinstance(other, Grad) else Grad(
            other, required_grad=False)

        out = Grad(
            val=self.val + other.val,
            required_grad=self.requires_grad or other.requires_grad,
            children=(self, other)
        )

        def out_backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = out_backward

        return out

    def __mul__(self, other) -> Grad:
        other = other if isinstance(other, Grad) else Grad(
            other, required_grad=False)

        out = Grad(
            val=self.val * other.val,
            required_grad=self.requires_grad or other.requires_grad,
            children=(self, other)
        )

        def out_backward():
            if self.requires_grad:
                self.grad += other.val * out.grad
            if other.requires_grad:
                other.grad += self.val * out.grad

        out._backward = out_backward

        return out

    def __pow__(self, other) -> Grad:  # self ^ other
        other = other if isinstance(other, Grad) else Grad(
            other, required_grad=False)

        out = Grad(
            val=float(self.val) ** other.val,
            required_grad=self.requires_grad or other.requires_grad,
            children=(self, other)
        )

        def out_backward():
            if self.requires_grad:
                self.grad += other.val * \
                    (self.val ** (other.val - 1)) * out.grad
            if other.requires_grad:
                other.grad += np.log(self.val) * \
                    (self.val ** other.val) * out.grad
        out._backward = out_backward

        return out

    def __radd__(self, other) -> Grad:
        return self + other

    def __rmul__(self, other) -> Grad:
        return self * other

    def __rpow__(self, other) -> Grad:  # other ^ self
        other = other if isinstance(other, Grad) else Grad(
            other, required_grad=False)
        return other ** self

    def __neg__(self) -> Grad:
        return self * -1

    def __sub__(self, other) -> Grad:
        return self + (-other)

    def __rsub__(self, other) -> Grad:
        return other + (-self)

    def __truediv__(self, other) -> Grad:
        other = other if isinstance(other, Grad) else Grad(
            other, required_grad=False)
        return self * (other ** -1)

    def __rtruediv__(self, other) -> Grad:
        return other * (self ** -1)

    def __lt__(self, other):
        return self.val < other.val if isinstance(other, Grad) else self.val < other

    def __le__(self, other):
        return self.val <= other.val if isinstance(other, Grad) else self.val <= other

    def __gt__(self, other):
        return self.val > other.val if isinstance(other, Grad) else self.val > other

    def __ge__(self, other):
        return self.val >= other.val if isinstance(other, Grad) else self.val >= other

    def __eq__(self, value):
        return self.val == value.val if isinstance(value, Grad) else self.val == value

    def __ne__(self, value):
        return self.val != value.val if isinstance(value, Grad) else self.val != value

    def __repr__(self) -> str:
        return f"Grad(value = {self.val}, grad={self.grad})"

    @staticmethod
    def full(shape: tuple | int, val, requires_grad=True):
        len = shape if isinstance(shape, int) else np.prod(shape)
        a = [Grad(val, requires_grad) for _ in range(len)]
        return np.array(a).reshape(shape)

    @staticmethod
    def rand(shape: tuple | int, requires_grad=True):
        len = shape if isinstance(shape, int) else np.prod(shape)
        a = [Grad(np.random.rand(), requires_grad) for _ in range(len)]
        return np.array(a).reshape(shape)

    @staticmethod
    def zeros(shape: tuple | int, requires_grad=True):
        return Grad.full(shape, 0, requires_grad)

    @staticmethod
    def ones(shape: tuple | int, requires_grad=True):
        return Grad.full(shape, 1, requires_grad)

    @staticmethod
    def array(arr, requires_grad=True):
        arr = np.array(arr)
        a = [Grad(x, requires_grad) for x in arr.flatten()]
        return np.array(a).reshape(arr.shape)

    @staticmethod
    def _exp(x: Grad) -> Grad:
        out = Grad(
            val=np.exp(x.val),
            required_grad=x.requires_grad,
            children=(x,)
        )

        def out_backward():
            if x.requires_grad:
                x.grad += out.val * out.grad

        out._backward = out_backward

        return out

    @staticmethod
    @overload
    def exp(x: Grad) -> Grad: ...

    @staticmethod
    @overload
    def exp(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def exp(x: Grad | np.ndarray) -> Grad | np.ndarray:
        if isinstance(x, Grad):
            return Grad._exp(x)
        return np.vectorize(Grad._exp)(x)

    @staticmethod
    def _log(x: Grad) -> Grad:
        out = Grad(
            val=np.log(x.val),
            required_grad=x.requires_grad,
            children=(x,)
        )

        def out_backward():
            if x.requires_grad:
                x.grad += (1 / x.val) * out.grad

        out._backward = out_backward

        return out

    @staticmethod
    @overload
    def log(x: Grad) -> Grad: ...

    @staticmethod
    @overload
    def log(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def log(x: Grad | np.ndarray) -> Grad | np.ndarray:
        if isinstance(x, Grad):
            return Grad._log(x)
        return np.vectorize(Grad._log)(x)

    @staticmethod
    def _relu(x: Grad) -> Grad:
        out = Grad(
            val=x.val if x.val > 0 else 0,
            required_grad=x.requires_grad,
            children=(x,)
        )

        def out_backward():
            if x.requires_grad:
                x.grad += (1 if x.val > 0 else 0) * out.grad

        out._backward = out_backward

        return out

    @staticmethod
    @overload
    def relu(x: Grad) -> Grad: ...

    @staticmethod
    @overload
    def relu(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def relu(x: Grad | np.ndarray) -> Grad | np.ndarray:
        if isinstance(x, Grad):
            return Grad._relu(x)
        return np.vectorize(Grad._relu)(x)

    @staticmethod
    def _clip(x: Grad, min_val, max_val) -> Grad:
        out = Grad(
            val=min(max(x.val, min_val), max_val),
            required_grad=x.requires_grad,
            children=(x,)
        )

        def out_backward():
            if x.requires_grad:
                x.grad += (min_val <= x.val <= max_val) * out.grad

        out._backward = out_backward

        return out

    @staticmethod
    @overload
    def clip(x: Grad, min_val, max_val) -> Grad: ...

    @staticmethod
    @overload
    def clip(x: np.ndarray, min_val, max_val) -> np.ndarray: ...

    @staticmethod
    def clip(x: Grad | np.ndarray, min_val, max_val) -> Grad | np.ndarray:
        if isinstance(x, Grad):
            return Grad._clip(x, min_val, max_val)
        return np.vectorize(lambda x: Grad._clip(x, min_val, max_val))(x)