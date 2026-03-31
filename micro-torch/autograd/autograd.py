from __future__ import annotations
import math
from typing import overload
import numpy as np


class Grad():
    def __init__(self, val, required_grad=True, children=()):
        self.val: float = val
        self.grad: float | None = 0 if required_grad else None
        self.requires_grad = required_grad

        self._children: tuple[Grad] = children
        self._backward = lambda: None

    def backprop(self, is_result=True):
        """This function computed the partial derivatives of the result to every involved input (which requires gradient)"""

        assert self.requires_grad, "Must require gradient"

        if is_result:
            self.grad = 1

        self._backward()
        for child in self._children:
            if child.requires_grad:
                child.backprop(is_result=False)

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
    def zeros(shape: tuple | int, requires_grad=True):
        return Grad.full(shape, 0, requires_grad)

    @staticmethod
    def ones(shape: tuple | int, requires_grad=True):
        return Grad.full(shape, 1, requires_grad)

    @staticmethod
    @overload
    def exp(x: Grad) -> Grad: ...

    @staticmethod
    @overload
    def exp(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def exp(x: Grad | np.ndarray) -> Grad | np.ndarray:
        return math.e ** x

    @staticmethod
    @overload
    def log(x: Grad, base: float = math.e) -> Grad: ...

    @staticmethod
    @overload
    def log(x: np.ndarray, base: float = math.e) -> np.ndarray: ...

    @staticmethod
    def log(x: Grad | np.ndarray, base: float = math.e) -> Grad | np.ndarray:
        if isinstance(x, Grad):
            out = Grad(
                val=math.log(x.val, base),
                required_grad=x.requires_grad,
                children=(x,)
            )

            def out_backward():
                if x.requires_grad:
                    x.grad += (1 / (x.val * math.log(base))) * out.grad

            out._backward = out_backward

            return out

        return np.vectorize(Grad.log)(x)

    @staticmethod
    @overload
    def relu(x: Grad) -> Grad: ...

    @staticmethod
    @overload
    def relu(x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def relu(x: Grad | np.ndarray) -> Grad | np.ndarray:
        mult = x > 0
        return mult * x