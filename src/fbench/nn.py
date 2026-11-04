"""From-scratch NumPy neural networks with hand-derived gradients.

Only NumPy is used for the model maths. Every backward pass is written by hand
so that it can be checked against central finite differences in E1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _gelu(z: np.ndarray) -> np.ndarray:
    u = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)
    return 0.5 * z * (1.0 + np.tanh(u))


def _gelu_grad(z: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    u = c * (z + 0.044715 * z**3)
    t = np.tanh(u)
    du = c * (1.0 + 3.0 * 0.044715 * z**2)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t**2) * du


ACTIVATIONS: dict[str, tuple] = {
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "relu": (lambda z: np.maximum(0.0, z), lambda z: (z > 0.0).astype(z.dtype)),
    "leaky_relu": (
        lambda z: np.where(z > 0.0, z, 0.01 * z),
        lambda z: np.where(z > 0.0, 1.0, 0.01),
    ),
    "gelu": (_gelu, _gelu_grad),
}


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def _he(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key, param in params.items():
            grad = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grad * grad
            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class TrainHistory:
    train_loss: list[float]
    train_accuracy: list[float]


class MLPClassifier:
    """One-hidden-layer MLP with hand-derived backpropagation."""

    def __init__(
        self,
        n_features: int,
        hidden: int,
        n_classes: int,
        activation: str = "relu",
        seed: int = 0,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        rng = np.random.default_rng(seed)
        self.activation_name = activation
        self.fn, self.dfn = ACTIVATIONS[activation]
        if activation in {"relu", "leaky_relu", "gelu"}:
            self.params = {
                "W1": _he(rng, n_features, hidden),
                "b1": np.zeros((1, hidden)),
                "W2": _he(rng, hidden, n_classes),
                "b2": np.zeros((1, n_classes)),
            }
        else:
            self.params = {
                "W1": _xavier(rng, n_features, hidden),
                "b1": np.zeros((1, hidden)),
                "W2": _xavier(rng, hidden, n_classes),
                "b2": np.zeros((1, n_classes)),
            }

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = self.fn(z1)
        logits = a1 @ p["W2"] + p["b2"]
        cache = {"X": X, "z1": z1, "a1": a1, "probs": softmax(logits)}
        return cache["probs"], cache

    def loss_and_grads(
        self, X: np.ndarray, y: np.ndarray, l2: float = 0.0
    ) -> tuple[float, dict[str, np.ndarray]]:
        probs, cache = self.forward(X)
        n = len(y)
        loss = cross_entropy(probs, y)
        if l2:
            loss += (
                0.5
                * l2
                * (float(np.sum(self.params["W1"] ** 2)) + float(np.sum(self.params["W2"] ** 2)))
            )
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        grads: dict[str, np.ndarray] = {}
        grads["W2"] = cache["a1"].T @ dlogits + l2 * self.params["W2"]
        grads["b2"] = dlogits.sum(axis=0, keepdims=True)
        da1 = dlogits @ self.params["W2"].T
        dz1 = da1 * self.dfn(cache["z1"])
        grads["W1"] = X.T @ dz1 + l2 * self.params["W1"]
        grads["b1"] = dz1.sum(axis=0, keepdims=True)
        return loss, grads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        lr: float = 0.01,
        batch_size: int = 128,
        l2: float = 1e-4,
        seed: int = 0,
    ) -> TrainHistory:
        rng = np.random.default_rng(seed + 991)
        optimizer = Adam(self.params, lr=lr)
        history = TrainHistory([], [])
        n = len(y)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                _, grads = self.loss_and_grads(X[idx], y[idx], l2=l2)
                optimizer.step(self.params, grads)
            probs, _ = self.forward(X)
            history.train_loss.append(cross_entropy(probs, y))
            history.train_accuracy.append(float(np.mean(probs.argmax(axis=1) == y)))
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ------------------------------------------------------- flat parameters
    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.params[k].ravel() for k in ("W1", "b1", "W2", "b2")])

    def set_flat_params(self, flat: np.ndarray) -> None:
        offset = 0
        for key in ("W1", "b1", "W2", "b2"):
            shape = self.params[key].shape
            size = int(np.prod(shape))
            self.params[key] = flat[offset : offset + size].reshape(shape).copy()
            offset += size

    def flat_grads(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
        _, grads = self.loss_and_grads(X, y, l2=l2)
        return np.concatenate([grads[k].ravel() for k in ("W1", "b1", "W2", "b2")])


class LinearRegressor:
    """Linear regression trained by gradient descent, used in the live-data test."""

    def __init__(self, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=n_features)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-3):
        pred = self.predict(X)
        residual = pred - y
        loss = float(np.mean(residual**2) + 0.5 * l2 * np.sum(self.w**2))
        n = len(y)
        dw = (2.0 / n) * (X.T @ residual) + l2 * self.w
        db = float(2.0 * np.mean(residual))
        return loss, dw, db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        lr: float = 0.01,
        l2: float = 1e-3,
    ) -> list[float]:
        m_w = np.zeros_like(self.w)
        v_w = np.zeros_like(self.w)
        m_b = 0.0
        v_b = 0.0
        losses: list[float] = []
        for step in range(1, epochs + 1):
            loss, dw, db = self.loss_and_grads(X, y, l2=l2)
            m_w = 0.9 * m_w + 0.1 * dw
            v_w = 0.999 * v_w + 0.001 * dw * dw
            m_b = 0.9 * m_b + 0.1 * db
            v_b = 0.999 * v_b + 0.001 * db * db
            m_w_hat = m_w / (1.0 - 0.9**step)
            v_w_hat = v_w / (1.0 - 0.999**step)
            m_b_hat = m_b / (1.0 - 0.9**step)
            v_b_hat = v_b / (1.0 - 0.999**step)
            self.w -= lr * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            losses.append(loss)
        return losses


def numerical_gradient_check(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_checks: int = 24,
    eps: float = 1e-5,
    seed: int = 0,
    l2: float = 0.0,
) -> dict[str, float]:
    """Central-difference check on a random subset of all parameters."""
    base = model.get_flat_params()
    analytic = model.flat_grads(X, y, l2=l2)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(base), size=min(n_checks, len(base)), replace=False)
    numeric = np.empty(len(indices))
    for j, idx in enumerate(indices):
        plus = base.copy()
        minus = base.copy()
        plus[idx] += eps
        minus[idx] -= eps
        model.set_flat_params(plus)
        loss_plus, _ = model.loss_and_grads(X, y, l2=l2)
        model.set_flat_params(minus)
        loss_minus, _ = model.loss_and_grads(X, y, l2=l2)
        numeric[j] = (loss_plus - loss_minus) / (2.0 * eps)
    model.set_flat_params(base)
    a = analytic[indices]
    denom = np.maximum(1.0, np.abs(a) + np.abs(numeric))
    rel = np.abs(a - numeric) / denom
    return {
        "n_checked": int(len(indices)),
        "max_abs_error": float(np.max(np.abs(a - numeric))),
        "max_relative_error": float(np.max(rel)),
        "mean_relative_error": float(np.mean(rel)),
        "cosine_similarity": float(
            np.dot(a, numeric) / (np.linalg.norm(a) * np.linalg.norm(numeric) + 1e-12)
        ),
    }
