import numpy as np

EPSILON = 1e-9  # For numerical stability in the log space
MIN_BIN_WIDTH = 1e-5

# Common functions
sig = lambda z: 1 / (1 + np.exp(-z))
sm = lambda z: np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]


def normalize_probabilities(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x) * 1 / x.shape[1] # Equal probability between classes
    total_probs = np.sum(x, axis=1, keepdims=True)
    np.divide(x, total_probs, out=out, where=total_probs!=0)
    return out

