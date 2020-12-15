from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from scipy.optimize import least_squares
import numpy as np
from utils import EPSILON


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
        self.regression.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)


class GammaRegression(Regression):
    """
    Fits the points to the curve y = (x/k)^(1/gamma)
    """

    k: float
    gamma: float

    def fit(self, x: np.ndarray, y: np.ndarray):
        def get_residuals(coefficients: np.ndarray):
            k, gamma = coefficients
            residuals = ((x.flatten() / k) ** (1 / gamma)) - y

            return residuals

        initial_coefficients = [0.5, 0.5]
        regression = least_squares(get_residuals, initial_coefficients, jac='3-point')
        self.k, self.gamma = regression.x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.power(x.flatten() / self.k, 1 / self.gamma)
