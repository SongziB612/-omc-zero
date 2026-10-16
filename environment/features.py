"""Canonical feature library shared by the environment (target generation)
and the agents (hypothesis space).

The library is complete-but-large: every target term is in the library,
but the library also contains many distractors, so sample efficiency
(and hence the value of shared validated terms) is the bottleneck.
"""
import numpy as np

FEATURES = {
    "1":                lambda X: np.ones(X.shape[0]),
    "x1":               lambda X: X[:, 0],
    "x2":               lambda X: X[:, 1],
    "x1^2":             lambda X: X[:, 0] ** 2,
    "x2^2":             lambda X: X[:, 1] ** 2,
    "x1*x2":            lambda X: X[:, 0] * X[:, 1],
    "x1^3":             lambda X: X[:, 0] ** 3,
    "x2^3":             lambda X: X[:, 1] ** 3,
    "x1^2*x2":          lambda X: X[:, 0] ** 2 * X[:, 1],
    "x1*x2^2":          lambda X: X[:, 0] * X[:, 1] ** 2,
    "sin(x1)":          lambda X: np.sin(X[:, 0]),
    "cos(x1)":          lambda X: np.cos(X[:, 0]),
    "sin(x2)":          lambda X: np.sin(X[:, 1]),
    "cos(x2)":          lambda X: np.cos(X[:, 1]),
    "sin(2*x1)":        lambda X: np.sin(2 * X[:, 0]),
    "cos(2*x1)":        lambda X: np.cos(2 * X[:, 0]),
    "sin(2*x2)":        lambda X: np.sin(2 * X[:, 1]),
    "cos(2*x2)":        lambda X: np.cos(2 * X[:, 1]),
    "sin(3*x1)":        lambda X: np.sin(3 * X[:, 0]),
    "cos(3*x1)":        lambda X: np.cos(3 * X[:, 0]),
    "sin(3*x2)":        lambda X: np.sin(3 * X[:, 1]),
    "cos(3*x2)":        lambda X: np.cos(3 * X[:, 1]),
    "sin(x1+x2)":       lambda X: np.sin(X[:, 0] + X[:, 1]),
    "cos(x1+x2)":       lambda X: np.cos(X[:, 0] + X[:, 1]),
    "sin(x1-x2)":       lambda X: np.sin(X[:, 0] - X[:, 1]),
    "cos(x1-x2)":       lambda X: np.cos(X[:, 0] - X[:, 1]),
    "sin(x1)*cos(x2)":  lambda X: np.sin(X[:, 0]) * np.cos(X[:, 1]),
    "cos(x1)*sin(x2)":  lambda X: np.cos(X[:, 0]) * np.sin(X[:, 1]),
}

FEATURE_NAMES = list(FEATURES.keys())
NAME_TO_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
TRIG_NAMES = [n for n in FEATURE_NAMES if n.startswith(("sin", "cos"))]


def design_matrix(X, feature_names):
    """X: (n, 2) -> (n, len(feature_names))."""
    X = np.asarray(X, dtype=float)
    cols = [FEATURES[n](X) for n in feature_names]
    return np.column_stack(cols)


def eval_expression(X, support, coef):
    """Evaluate a linear combination given support names and coefficients."""
    Phi = design_matrix(X, support)
    return Phi @ np.asarray(coef, dtype=float)
