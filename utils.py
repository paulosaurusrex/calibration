import numpy as np

EPSILON = 1e-9  # For numerical stability in the log space

# Common functions
sig = lambda z: 1 / (1 + np.exp(-z))
sm = lambda z: np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]
normalize_probabilities = lambda x: x / np.sum(x, axis=1, keepdims=True)

