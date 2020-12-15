from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from scipy.optimize import least_squares, minimize
import numpy as np


class Regression(ABC):

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def predict(self, x: np.ndarray):
        ...


class LinearRegression(Regression):

    def __init__(self):
        self.regression = SklearnLinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.regression.fit(x.reshape(-1, 1), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.regression.predict(x.reshape(-1, 1))


class GammaRegression(Regression):
    """
    Fits the points to the curve y = (x/k)^(1/gamma)
    """

    k: float
    gamma: float

    def fit(self, x: np.ndarray, y: np.ndarray):
        def get_residuals(coefficients: np.ndarray):
            k, gamma = coefficients
            residuals = ((x / k) ** (1 / gamma)) - y

            return residuals

        initial_guess = [0.5, 0.5]
        regression = least_squares(get_residuals, initial_guess, jac='3-point')
        self.k, self.gamma = regression.x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.power(x / self.k, 1 / self.gamma)


class ConstrainedLinearRegression(Regression):
    """
    Fits a line to the points ensuring that the slope is positive, f(1) <= 1 and f(0) >= 0.
    """

    a: float
    b: float

    def fit(self, x: np.ndarray, y: np.ndarray):
        def obj_fun(coefficients: np.ndarray, design_matrix: np.ndarray, y: np.ndarray):
            y_hat = design_matrix.dot(coefficients.reshape(-1, 1))
            return np.sum((y_hat - y) ** 2)

        # y = ax + b
        # a >= 0 (non-negative slope) and b >=0 (y >= 0)
        bounds = [(0, None), (0, None)]
        # y <= 1
        constraints = {'type': 'ineq', 'fun': lambda z: -np.sum(z) + 1}

        design_matrix = np.ones((x.shape[0], 2))
        design_matrix[:, 0] = x
        initial_guess = [0.5, 0.5]
        regression = minimize(obj_fun, args=(design_matrix, y.reshape(-1, 1)), x0=initial_guess, bounds=bounds,
                 constraints=constraints)
        self.a, self.b = regression.x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.a * x + self.b
