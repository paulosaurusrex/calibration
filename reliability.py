from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from matplotlib.axes import Axes
from typing import List, Tuple
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import math
from utils import MIN_BIN_WIDTH


class Kernel(ABC):
    @abstractmethod
    def pdf(self, probs: np.ndarray, bin_center: float, bin_width: float,
            probs_in_bin: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def plot(self, bin_center: float, bin_height: float, bin_width: float,
             probs_in_bin: np.ndarray, ax: Axes, n: int = 50):
        pass

    def format_weight(self, count):
        return str(count)


class DeltaKernel(Kernel):
    """Class to represent a convolutional delta filter."""

    def pdf(self, probs: np.ndarray, bin_center: float, bin_width: float,
            probs_in_bin: np.ndarray) -> Tuple[np.ndarray, float]:
        # 1 if a point is inside the bin, 0 otherwise
        left_prob = bin_center - bin_width / 2
        right_prob = bin_center + bin_width / 2
        if math.isclose(right_prob, 1):
            pdfs = np.array((probs >= left_prob) & (
                    probs <= right_prob), dtype=np.int)
        else:
            pdfs = np.array((probs >= left_prob) & (
                    probs < right_prob), dtype=np.int)

        mean = np.mean(probs_in_bin)

        return pdfs, mean

    def plot(self, bin_center: float, bin_height: float, bin_width: float,
             probs_in_bin: np.ndarray, ax: Axes, n: int = 50):
        if len(probs_in_bin) > 0:
            left_prob = bin_center - bin_width / 2
            right_prob = bin_center + bin_width / 2
            x = np.linspace(left_prob, right_prob, n)
            y = np.ones(n) * bin_height
            ax.plot(x, y)


class GaussianKernel(Kernel):
    """Class to represent a convolutional Gaussian filter."""
    SMALLEST_STD = MIN_BIN_WIDTH

    def __init__(self, std: float = None):
        if std is not None and std <= 0:
            raise Exception('STD must be greater than 0.')
        self.std = std

    def pdf(self, probs: np.ndarray, bin_center: float, bin_width: float,
            probs_in_bin: np.ndarray) -> Tuple[np.ndarray, float]:

        mean, std = self.__get_mean_and_std(bin_center, bin_width, probs_in_bin)
        if std == 0:
            # This can happen if bin_width == 0 in the adaptive case
            pdfs = norm.pdf(probs, mean, GaussianKernel.SMALLEST_STD)
        else:
            pdfs = norm.pdf(probs, mean, std)

        return pdfs, mean

    def plot(self, bin_center: float, bin_height: float, bin_width: float,
             probs_in_bin: np.ndarray, ax: Axes, n: int = 100):

        mean, std = self.__get_mean_and_std(bin_center, bin_width, probs_in_bin)

        if std != 0:
            min_y = -2

            left_prob = mean - 2 * std
            right_prob = mean + 2 * std
            x = np.linspace(left_prob, right_prob, n)
            y = norm.pdf(x, mean, std)
            y = y - np.max(y) + bin_height

            if len(probs_in_bin) == 0:
                ax.plot(x[y >= min_y], y[y >= min_y], 'm--', alpha=0.5)
            else:
                ax.plot(x[y >= min_y], y[y >= min_y], 'm', alpha=0.5)

    def __get_mean_and_std(self, bin_center: float, bin_width: float,
                           probs_in_bin: np.ndarray):
        # If an standard deviation was provided, use that fix one, otherwise
        # use the probabilities in the bin or the bin width
        if self.std is None:
            # Adaptive standard deviation
            if len(probs_in_bin) > 1:
                std = np.std(probs_in_bin)
            else:
                # The standard deviation is approximately 1/4
                # of the range of the data
                std = bin_width / 4
        else:
            std = self.std

        # Adaptive mean
        if len(probs_in_bin) == 0:
            mean = 0
        else:
            # The average probability in a bin with Gaussian kernel is a
            # weighted average of the probabilities in the bin according to a
            # normal distribution with mean and variance defined by the center
            # and width of the bin.
            pdfs = norm.pdf(probs_in_bin, bin_center, bin_width / 4)
            mean = np.sum(probs_in_bin * pdfs) / np.sum(pdfs)

        return mean, std

    def format_weight(self, count):
        return '{:.2f}'.format(count)


@dataclass
class ReliabilityCurve:
    """Class to represent a reliability curve"""

    # Probabilities in the center of each bin
    bin_centers: np.ndarray

    # Heights of each bin (accuracy)
    bin_heights: np.ndarray

    # Weighted average probabilities of the points in each bin. The weight is the pdf defined by the
    # kernel used to generate the curve.
    bin_avg_probs: np.ndarray

    # Weights of the points in each bin. It's equivalent to the number of
    # points in each bin if the Delta kernel is used for constructing the curve.
    bin_weights: np.ndarray

    # Width of each bin
    bin_widths: np.ndarray

    # Kernel used to construct the curve
    kernel: Kernel

    # List of probabilities per bin
    probs_per_bin: List[np.ndarray]

    # Whether the curve was generated with bins of equivalent mass
    even_mass: bool

    def __post_init__(self):
        self.bin_counts = [len(probs_in_bin) for probs_in_bin in
                           self.probs_per_bin]

    def plot(self, fig, ax, use_bin_centers=False, show_bars=False,
             show_counts=False, show_kernel=False, show_colormap=False):
        x = self.bin_centers if use_bin_centers else self.bin_avg_probs
        y = self.bin_heights

        if show_bars:
            ax.bar(self.bin_centers, y, width=self.bin_widths,
                   color='c', edgecolor='k', alpha=0.1)
        ax.plot(x, y, color='black', alpha=0.2)

        # Perfect calibration
        ax.plot([0, 1], [0, 1], color='FireBrick', linestyle='--',
                alpha=0.5,
                label="Perfect calibration")

        if show_counts:
            if show_bars:
                # If bars are shown, align the text with the center of the bar
                # instead of showing them over the points
                x = self.bin_centers

            for i, count in enumerate(self.bin_counts):
                if count > 0:
                    # Only show counts > 0
                    ax.text(x[i], y[i] + 0.05, count, ha='center', rotation=45)

        if show_kernel:
            # Shows a pirce of the kernel distribution centered at each point
            for bin_center, bin_height, bin_width, probs_in_bin in zip(
                    self.bin_centers,
                    self.bin_heights,
                    self.bin_widths,
                    self.probs_per_bin):
                self.kernel.plot(bin_center, bin_height,
                                 bin_width, probs_in_bin, ax)

        if show_colormap:
            viridis = cm.get_cmap('viridis', max(self.bin_counts))
            ax.scatter(x, y, color=viridis(self.bin_counts))
            sm = cm.ScalarMappable(cmap=viridis,
                                   norm=plt.Normalize(0, max(self.bin_counts)))
            sm.set_array(np.empty([]))
            cb = fig.colorbar(sm, ax=ax, label="Number of trials in bin")
            cb.ax.yaxis.set_major_locator(
                FixedLocator(cb.ax.get_yticks().tolist()))
            cb.ax.set_yticklabels(["{:.0f}".format(i) for i in cb.get_ticks()])
        else:
            ax.scatter(x, y, color='black')

        ax.set_xlabel('Probability')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(-0.025, 1.025 + int(show_counts) * 0.05)
        ax.set_ylim(-0.025, 1.025 + int(show_counts) * 0.05)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

    def get_bin_limits(self, bin_idx: int) -> Tuple[float, float]:
        left_prob = self.bin_centers[bin_idx] - self.bin_widths[bin_idx] / 2
        right_prob = self.bin_centers[bin_idx] + self.bin_widths[bin_idx] / 2

        return left_prob, right_prob

    def ece(self, use_bin_centers=False):
        """
        Computes the Estimated Calibration Error (ECE)
        :param use_bin_centers: use the distance of the bin centers to the
        perfect calibration line instead of the average probability of the
        points in the bin.
        :return: ECE
        """

        x = self.bin_centers if use_bin_centers else self.bin_avg_probs
        return np.sum(np.abs(self.bin_heights - x)) / len(x)


class ReliabilityFit:
    """Class to represent a fitting procedure to build reliability curves."""

    def __init__(self, probs: np.ndarray, classes: np.ndarray, kernel: Kernel,
                 even_mass=True):
        if len(set(classes)) > 2 or len(probs.shape) > 1:
            raise Exception(
                'Reliability curves are only defined '
                'for binary class problems.')

        self.probs = probs
        self.classes = classes
        self.kernel = kernel
        self.even_mass = even_mass

    def get_curve(self, num_points: int = 10) -> ReliabilityCurve:
        """Calculates points of interest to plot a reliability curve."""

        if self.even_mass:
            # The number of points in each bin is roughly the same and the
            # width of each bin is variable.
            sorted_data = sorted(zip(self.probs, self.classes))
            sorted_probs, sorted_classes = map(np.array, zip(*sorted_data))
            num_points_per_bin = int(np.ceil(len(sorted_probs) / num_points))

            # Guarantee at least two points per bin
            num_points_per_bin = np.max([2, num_points_per_bin])

            bin_centers = []
            bin_widths = []
            probs_per_bin = []
            last_bin_end = 0
            for i in range(0, len(sorted_probs), num_points_per_bin):
                right_idx = np.min(
                    [i + num_points_per_bin, len(sorted_probs) - 1])
                next_left_idx = np.min(
                    [i + num_points_per_bin + 1, len(sorted_probs) - 1])

                left_prob = last_bin_end
                right_prob = (sorted_probs[right_idx] + sorted_probs[
                    next_left_idx]) / 2
                bin_width = max(right_prob - left_prob, MIN_BIN_WIDTH)
                mid_prob = (right_prob - left_prob) / 2 + left_prob
                last_bin_end = right_prob

                bin_centers.append(mid_prob)
                bin_widths.append(bin_width)
                probs_per_bin.append(sorted_probs[i: right_idx])
        else:
            # Fixed width and number of points per bin variable
            bin_width = 1 / num_points
            bin_widths = [bin_width] * num_points
            bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2,
                                      num_points)
            probs_per_bin = []
            for bin_idx, bin_center in enumerate(bin_centers):
                left_prob = bin_center - bin_width / 2
                right_prob = bin_center + bin_width / 2

                if bin_idx == len(bin_centers) - 1:
                    probs_in_bin = self.probs[
                        (left_prob <= self.probs) & (self.probs <= right_prob)]
                else:
                    probs_in_bin = self.probs[
                        (left_prob <= self.probs) & (self.probs < right_prob)]
                probs_per_bin.append(probs_in_bin)

        bin_weights = []
        bin_centers_with_mass = []
        bin_avg_probs = []
        bin_heights = []
        bin_widths_with_mass = []
        probs_per_bin_with_mass = []

        for i, (bin_center, bin_width, probs_in_bin) in enumerate(
                zip(bin_centers, bin_widths, probs_per_bin)):

            if len(probs_in_bin) > 0:
                pdfs, avg_prob_in_bin = self.kernel.pdf(self.probs, bin_center,
                                                        bin_width,
                                                        probs_in_bin)
                normalizer = np.sum(pdfs)
                if normalizer == 0:
                    continue

                accuracy = np.sum(self.classes * pdfs) / normalizer

                bin_centers_with_mass.append(bin_center)
                bin_avg_probs.append(avg_prob_in_bin)
                bin_heights.append(accuracy)
                bin_weights.append(normalizer)
                bin_widths_with_mass.append(bin_width)
                probs_per_bin_with_mass.append(probs_in_bin)

        return ReliabilityCurve(np.array(bin_centers_with_mass),
                                np.array(bin_heights),
                                np.array(bin_avg_probs),
                                np.array(bin_weights),
                                np.array(bin_widths_with_mass),
                                self.kernel,
                                probs_per_bin_with_mass,
                                self.even_mass)

# if __name__ == '__main__':
# import json
# import functions as f
# import matplotlib.pylab as plt
#
#
# def get_probs_and_classes(eval_dir, data_dir):
#     evaluations = json.load(open(eval_dir, 'r'))
#     true_q_data = np.loadtxt(data_dir, dtype=np.int)[:, -1]
#
#     true_values = f.get_true_values_correct_order(evaluations, true_q_data)
#     probs = f.get_probabilities_and_true_labels(evaluations, [0, 1, 2],
#                                                 'TrainingCondition', 0, 1,
#                                                 -1)
#     probs = np.column_stack([probs[0], probs[1], probs[2]])
#
#     return probs, true_values
#
#
# val_probs, val_classes = get_probs_and_classes(
#     '../calibration/evaluations/2e_val/evaluations.json',
#     '../calibration/evaluations/2e_val/TrainingCondition')
# # test_probs, test_classes = get_probs_and_classes(
# #     'evaluations/2e_test/evaluations.json',
# #     'evaluations/2e_test/TrainingCondition')
#
# curve = ReliabilityFit(val_probs[:, 0],
#                        np.array(val_classes == 0, dtype=int),
#                        NormalKernel(), even_mass=True).get_curve(100)
#
# plt.rcParams["figure.figsize"] = 10, 10
# fig = plt.figure()
# ax = fig.gca()
# curve.plot(ax, use_avg_points=True, show_bars=True, show_counts=True)
