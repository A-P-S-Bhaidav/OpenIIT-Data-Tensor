"""Probability calibration.

``HierarchicalPlatt`` fits a global Platt scaling on out-of-fold scores and
then a per-segment (portfolio) Platt scaling that is L2-shrunk towards the
global parameters. Small portfolios therefore inherit the global mapping
while large portfolios get their own, which is what makes "0.6 mean the same
thing" for a salaried personal-loan borrower and a microfinance borrower
without overfitting thin segments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_platt(z: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    def loss(theta):
        a, b = theta
        p = np.clip(sigmoid(a * z + b), EPS, 1 - EPS)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return nll + lam * np.sum((theta - prior) ** 2)

    def grad(theta):
        a, b = theta
        p = sigmoid(a * z + b)
        r = p - y
        return np.array([np.sum(r * z), np.sum(r)]) + 2 * lam * (theta - prior)

    return minimize(loss, prior.copy(), jac=grad, method="L-BFGS-B").x


@dataclass
class HierarchicalPlatt:
    shrinkage: float = 25.0
    global_params: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    segment_params: Dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, p: np.ndarray, y: np.ndarray, segment: np.ndarray) -> "HierarchicalPlatt":
        z, y = logit(p), np.asarray(y, dtype=float)
        self.global_params = _fit_platt(z, y, np.array([1.0, 0.0]), lam=1e-3)
        segment = np.asarray(segment).astype(str)
        for s in np.unique(segment):
            m = segment == s
            self.segment_params[s] = _fit_platt(z[m], y[m], self.global_params, lam=self.shrinkage)
        return self

    def transform(self, p: np.ndarray, segment) -> np.ndarray:
        z = logit(p)
        seg = np.broadcast_to(np.asarray(segment).astype(str), z.shape)
        out = np.empty_like(z)
        for i, (zi, si) in enumerate(zip(z.ravel(), seg.ravel())):
            a, b = self.segment_params.get(si, self.global_params)
            out.ravel()[i] = sigmoid(a * zi + b)
        return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass ECE (robust for small segments)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    bins = np.array_split(order, min(n_bins, len(y)))
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)) / len(y))
