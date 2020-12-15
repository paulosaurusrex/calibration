from reliability import ReliabilityCurve, ReliabilityFit, DeltaKernel
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from regression import Regression, LinearRegression, ConstrainedLinearRegression, GammaRegression
from utils import EPSILON, sm, normalize_probabilities


class Calibration(ABC):
    """Class to represent a probability calibration method."""

    @abstractmethod
    def fit(self, probs: np.ndarray, classes: np.ndarray):
        pass

    @abstractmethod
    def transform(self, probs: np.ndarray, normalize=True) -> np.ndarray:
        pass

    @staticmethod
    def normalize(probs: np.ndarray):
        return probs / np.sum(probs, axis=1)[:, None]


class IsotonicCalibration(Calibration):

    regressions: List[IsotonicRegression] = []

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        self.regressions = []

        for i in range(probs.shape[1]):
            x = np.array(probs[:, i])
            y = np.array(classes == i, dtype=np.int)
            regression = IsotonicRegression(y_min=0, y_max=1)
            regression.fit(x, y)
            self.regressions.append(regression)

    def transform(self, probs: np.ndarray, normalize: bool = True):
        new_probs = np.empty_like(probs)

        for i, regression in enumerate(self.regressions):
            new_probs[:, i] = regression.predict(probs[:, i])

        if normalize:
            new_probs = self.normalize(new_probs)

        return new_probs


class PlattCalibration(Calibration):

    regressions: List[LogisticRegression] = []

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        self.regressions = []

        for i in range(probs.shape[1]):
            # Predictions are done individually per class because we want to let the user decide
            # if she wants to normalize the predicted probabilities in the transform method or not.
            # Using the multiclass fit from sklearn will always normalize the probabilities.
            x = np.array(probs[:, i])
            y = np.array(classes == i, dtype=np.int)
            regression = LogisticRegression()
            regression.fit(x.reshape(-1, 1), y)
            self.regressions.append(regression)

    def transform(self, probs: np.ndarray, normalize: bool = True):
        new_probs = np.empty_like(probs)

        for i, regression in enumerate(self.regressions):
            new_probs[:, i] = regression.predict_proba(probs[:, i].reshape(-1, 1))[:, 1]

        if normalize:
            new_probs = self.normalize(new_probs)

        return new_probs


class BetaCalibration(Calibration):
    regressions: List[LogisticRegression] = []

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        """
        We fit a logistic regression in the log space as proposed in
        Kull, M., Silva Filho, T. M., & Flach, P. (2017).
        Beyond Sigmoids: How to obtain well-calibrated probabilities
        from binary classifiers with beta calibration.
        """

        self.regressions = []
        for i in range(probs.shape[1]):
            x1_log = np.log(probs[:, i] + EPSILON)
            x2_log = -np.log(1 - probs[:, i] + EPSILON)
            x = np.array([x1_log, x2_log]).T
            y = np.array(classes == i, dtype=np.int)
            regression = LogisticRegression().fit(x, y)

            self.regressions.append(regression)

    def transform(self, probs: np.ndarray, normalize: bool = True):
        new_probs = np.empty_like(probs)

        for i, regression in enumerate(self.regressions):
            a = regression.coef_[0][0]
            b = regression.coef_[0][1]
            c = regression.intercept_[0]

            z = probs[:, i]
            p1 = np.ones_like(z) * EPSILON
            np.power(z, a, out=p1, where=(z != 0))
            p2 = np.ones_like(z) * EPSILON
            np.power(1 - z, b, out=p2, where=(1 - z != 0))
            p3 = np.exp(c)

            new_probs[:, i] = 1 / (1 + 1 / (p3 * (p1 / p2)))

        if normalize:
            new_probs = self.normalize(new_probs)

        return new_probs


class TemperatureScaling(Calibration):

    def __init__(self, temperature: float):
        self.temperature = temperature

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        pass

    def transform(self, probs: np.ndarray, normalize: bool = True):
        logits = np.log(probs / (1 - probs + EPSILON) + EPSILON)
        return sm(logits / self.temperature)


class BinningCalibration:

    curves: List[ReliabilityCurve]
    num_points: int = 10
    even_mass: bool = True

    def __init__(self, num_points: int = 10, even_mass: bool = True):
        self.num_points = num_points
        self.even_mass = even_mass

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        self.curves = []

        for i in range(probs.shape[1]):
            x = probs[:, i]
            y = np.array(classes == i, dtype=np.int)
            rel_fit = ReliabilityFit(x, y, DeltaKernel(), even_mass=self.even_mass)
            self.curves.append(rel_fit.get_curve(self.num_points))

    def transform(self, probs: np.ndarray, normalize: bool = True):
        new_probs = np.empty_like(probs)

        for i, curve in enumerate(self.curves):
            z = probs[:, i]
            sorted_z = sorted(zip(z, np.arange(len(z))))
            bin_idx = 0
            calibrated_z = np.zeros_like(z)
            for z_prob, idx in sorted_z:
                while not curve.is_in_bin(z_prob, bin_idx):
                    bin_idx += 1
                calibrated_z[idx] = curve.bin_heights[bin_idx]

            new_probs[:, i] = calibrated_z

        if normalize:
            new_probs = normalize_probabilities(new_probs)

        return new_probs


# Unbounded methods
class RegressionCalibration(ABC):

    regressions: List[Regression] = []

    def __init__(self, kernel, num_points: int = 10, even_mass: bool = True, clip_logits: bool = False):
        self.kernel = kernel
        self.num_points = num_points
        self.even_mass = even_mass
        self.clip_logits = clip_logits

    def fit(self, probs: np.ndarray, classes: np.ndarray):
        self.regressions = []

        for i in range(probs.shape[1]):
            x = probs[:, i]
            y = np.array(classes == i, dtype=np.int)
            rel_fit = ReliabilityFit(x, y, self.kernel, self.even_mass)
            curve = rel_fit.get_curve(self.num_points)

            regression = self.get_regression_method()
            regression.fit(curve.bin_avg_probs, curve.bin_heights)

            self.regressions.append(regression)

    def transform(self, probs: np.ndarray, normalize: bool = True):
        new_probs = np.empty_like(probs)

        for i, regression in enumerate(self.regressions):
            z = probs[:, i]
            logits = regression.predict(z)

            if self.clip_logits:
                logits[logits < 0] = 0
                logits[logits > 1] = 1

            new_probs[:, i] = logits

        return new_probs

    @abstractmethod
    def get_regression_method(self):
        ...


class LinearRegressionCalibration(RegressionCalibration):

    def get_regression_method(self):
        return LinearRegression()


class ConstrainedLinearRegressionCalibration(RegressionCalibration):

    def get_regression_method(self):
        return ConstrainedLinearRegression()


class GammaRegressionCalibration(RegressionCalibration):

    def get_regression_method(self):
        return GammaRegression()

