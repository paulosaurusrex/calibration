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
        if probs.shape[1] > 1:
            return probs / np.sum(probs, axis=1)[:, None]
        else:
            return probs


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
                left_prob, right_prob = curve.get_bin_limits(bin_idx)
                last_bin = bin_idx == len(curve.bin_centers) - 1

                if not last_bin:
                    while z_prob >= right_prob:
                        bin_idx += 1
                        left_prob, right_prob = curve.get_bin_limits(bin_idx)

                if z_prob < left_prob:
                    # There's no bin for the probability in the calibrated curve
                    calibrated_z[idx] = 0
                else:
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

        if normalize:
            new_probs = normalize_probabilities(new_probs)

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

if __name__ == '__main__':
    import numpy as np
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt

    def get_data(num_probs=1000):
        np.random.seed(42)
        X, y = datasets.make_classification(n_samples=6000, n_features=20, n_informative=3, n_redundant=17, n_classes=3)

        train_samples = 6000 - 2*num_probs  # Samples used for training the model
        X_train = X[:train_samples]
        X_test = X[train_samples:]
        y_train = y[:train_samples]
        y_test = y[train_samples:]

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        prob_pos = rfc.predict_proba(X_test)

        probs_train = prob_pos[:num_probs, :]  # Used to calibrate the curves
        classes_train = y_test[:num_probs]
        probs_val = prob_pos[num_probs:, :]  # Used to test the calibration
        classes_val = y_test[num_probs:]

        return probs_train, classes_train, probs_val, classes_val


    probs_train, classes_train, probs_val, classes_val = get_data(2000)
    from reliability import GaussianKernel

    curve = ReliabilityFit(probs_train[:, 0], classes_train == 0, GaussianKernel(), True).get_curve(10)

    binning = BetaCalibration()
    binning.fit(probs_train, classes_train)

    X = np.empty((100, 3))
    X[:, 0] = np.linspace(0, 1, 100)
    X[:, 1] = np.linspace(0, 1, 100)
    X[:, 2] = np.linspace(0, 1, 100)
    y = binning.transform(X, False)

    plt.plot(X[:, 2], y[:, 2])
    plt.show()
    # probs = binning.transform(probs_val, normalize=False)

