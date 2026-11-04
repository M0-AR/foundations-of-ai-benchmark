"""A minimal, independently written reverse-mode automatic differentiation engine.

The implementation is deliberately scalar and transparent: every operation
records a local derivative, and ``backward`` applies the chain rule in reverse
topological order. It exists to be *verified* against finite differences and
PyTorch, not to be fast.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


class Value:
    __slots__ = ("data", "grad", "_prev", "_backward", "label")

    def __init__(self, data: float, label: str = "", _children: Iterable[Value] = ()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._backward = lambda: None  # type: ignore[assignment]
        self.label = label

    # ------------------------------------------------------------------ ops
    def __add__(self, other: Any) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, "+", (self, other))

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Any) -> Value:
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __rsub__(self, other: Any) -> Value:
        return (other if isinstance(other, Value) else Value(other)) + (-self)

    def __mul__(self, other: Any) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, "*", (self, other))

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> Value:
        return self * other

    def __truediv__(self, other: Any) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other: Any) -> Value:
        return (other if isinstance(other, Value) else Value(other)) * (self**-1.0)

    def __pow__(self, exponent: float) -> Value:
        if isinstance(exponent, Value):
            raise TypeError("Value exponents are not supported")
        out = Value(self.data**exponent, f"**{exponent}", (self,))

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1.0)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), "exp", (self,))

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, "tanh", (self,))

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, "sigmoid", (self,))

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), "relu", (self,))

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), "log", (self,))

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # -------------------------------------------------------------- backward
    def backward(self) -> None:
        order: list[Value] = []
        visited: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                visit(child)
            order.append(node)

        visit(self)
        self.grad += 1.0
        for node in reversed(order):
            node._backward()

    def zero_grad(self) -> None:
        visited: set[int] = set()

        def visit(node: Value) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            node.grad = 0.0
            for child in node._prev:
                visit(child)

        visit(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self.label!r})"


def expression(a: float, b: float, return_inputs: bool = False):
    """A non-trivial test expression used by the gradient-fidelity experiment."""
    x = Value(a, "a")
    y = Value(b, "b")
    z = (x * y + x**2).tanh()
    positive = (z * z + 1.0) * (y**3).exp()
    out = positive.log() + (x / (y * y + 1.0)).sigmoid()
    if return_inputs:
        return out, x, y
    return out
