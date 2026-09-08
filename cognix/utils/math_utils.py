"""Numerical utilities for COGNIX."""
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, list]

# Numerical stability constant used throughout COGNIX
numerical_stability_eps: float = 1e-8


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + numerical_stability_eps)


def entropy(probs: np.ndarray, axis: int = -1, base: float = 2.0) -> np.ndarray:
    """Shannon entropy H(p) = -sum(p * log(p))."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, numerical_stability_eps, 1.0)
    return -np.sum(probs * np.log(probs + numerical_stability_eps) / np.log(base), axis=axis)


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """KL divergence D_KL(P || Q) = sum(P * log(P/Q))."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, numerical_stability_eps, 1.0)
    q = np.clip(q, numerical_stability_eps, 1.0)
    return np.sum(p * np.log(p / q), axis=axis)


def beta_mean(alpha: float, beta: float) -> float:
    """Mean of Beta(alpha, beta) distribution."""
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    """Variance of Beta(alpha, beta) distribution."""
    ab = alpha + beta
    return (alpha * beta) / (ab ** 2 * (ab + 1))
