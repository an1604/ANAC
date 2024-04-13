import random
import warnings

import numpy as np
from negmas import SAOState
from scipy.stats import pearsonr

from ANAC.learner import Learner


def correlation_to_probability(correlation_coefficient):
    return (1 + correlation_coefficient) / 2


def fill_with_mean(x, y):
    # Find the length of the longest array
    max_length = max(len(x), len(y))
    if len(x) < max_length:
        mean_value = np.mean(x)
        x = np.concatenate([x, np.full(max_length - len(x), mean_value)])
    else:
        mean_value = np.mean(y)
        y = np.concatenate([y, np.full(max_length - len(y), mean_value)])
    return x, y


def calc_corr(x, y):
    with warnings.catch_warnings(record=True) as w:
        pearson_corr, _ = pearsonr(x, y)
        if np.isnan(pearson_corr):
            return 0.1
        return pearson_corr


class DetectingRegion:
    def __init__(self, T: int, Nt: int, Np: int):
        self.T = T  # Deadline
        self.current_time = 0  # Current negotiation time
        self.Nt = Nt  # Number of columns
        self.Np = Np  # Number of rows
        self.n_cells = Nt * Np
        self.cells = []  # List to store detecting cells
        self.random_reservation_points = []  # List to store random reservation points
        self.regression_curves = []  # List to store regression curves
        self.fitted_offers = []  # List to store fitted offers
        self.correlations = []  # List to store non-linear correlations
        self.probs = []
        self.initialize_detecting_region()

    def initialize_detecting_region(self):
        # Define detecting region
        self.detecting_region = (0, self.T, 0, 100)  # Initial detecting region

        # Initialize detecting cells and generate random reservation points
        self.update_detecting_region()

    def update_detecting_region(self, state: SAOState = None, max_price=100):
        # Update the detecting region with the current negotiation time
        # TODO: current price as min price
        if state is not None:
            self.current_time = state.step
        else:
            self.current_time = 0
        max_price = max_price
        if self.current_time == 0:
            max_price = 100
        self.detecting_region = (self.current_time, self.T, 0, max_price)

        # Clear previous random reservation points and detecting cells
        self.random_reservation_points.clear()
        self.cells.clear()
        self.regression_curves.clear()
        self.fitted_offers.clear()
        self.correlations.clear()
        self.probs.clear()

        # Initialize detecting cells and generate random reservation points
        for t_idx in range(self.Nt):
            t_low = (
                    self.detecting_region[0]
                    + t_idx
                    * (self.detecting_region[1] - self.detecting_region[0])
                    / self.Nt
            )
            t_high = (
                    self.detecting_region[0]
                    + (t_idx + 1)
                    * (self.detecting_region[1] - self.detecting_region[0])
                    / self.Nt
            )
            for p_idx in range(self.Np):
                p_low = (
                        self.detecting_region[2]
                        + p_idx
                        * (self.detecting_region[3] - self.detecting_region[2])
                        / self.Np
                )
                p_high = (
                        self.detecting_region[2]
                        + (p_idx + 1)
                        * (self.detecting_region[3] - self.detecting_region[2])
                        / self.Np
                )
                tx = random.uniform(t_low, t_high)
                px = random.uniform(p_low, p_high)
                self.cells.append((t_low, t_high, p_low, p_high))
                self.random_reservation_points.append((tx, px))
        # Initialize the cell's prior probability
        for i in range(len(self.cells)):
            prob = Learner(prior=1 / self.n_cells)
            self.probs.append(prob)

    def print_detecting_region(self):
        print(f"Detecting region: {self.detecting_region}")
        for idx, cell in enumerate(self.cells):
            print(f"Cell {idx}: {cell}")
        for idx, point in enumerate(self.random_reservation_points):
            print(f"Random reservation point {idx}: {point}")

    def generate_regression_curve(self, history: list[float]):
        print("Generating random regression curve")
        # init price
        init_price = history[0] if history[0] != 0 else 100

        for reservation_point in self.random_reservation_points:
            # claculate the beta coefficient
            beta = 0
            up = 0
            down = 0
            t_i_x, p_i_x = reservation_point[0], reservation_point[1]
            # print(f"t_i_x: {t_i_x}")
            # print(f"p_i_x: {p_i_x}")

            for i in range(1, self.current_time):
                history[i] = history[i] if history[i] != init_price else init_price - 1
                p_star_i = np.log((init_price - history[i]) / (init_price - p_i_x))
                t_star_i = np.log(i / t_i_x)
                t_star_i_current = np.log(self.current_time / t_i_x)
                up += (p_star_i - p_i_x) * (t_star_i - t_i_x)
                down += (t_star_i - t_i_x) ** 2

            beta = up / down
            self.regression_curves.append((init_price, p_i_x, t_i_x, beta))

    def clalculate_fitted_offers(self):
        for curve in self.regression_curves:
            index = 0
            offer_list = []  # List to store fitted offers
            init_price, p_i_x, t_i_x, beta = curve
            for i in range(self.current_time):
                offer = init_price + (p_i_x - init_price) * (i / t_i_x) ** beta
                offer_list.append(offer)
                # print(f"for cell: {index} at time: {i} offer: {offer}")
            index += 1
            self.fitted_offers.append(offer_list)

    def get_non_linear_correlation(self, history: list[float]):
        # Calculate the correlation coefficient
        index = 0
        for fitted_offer in self.fitted_offers:
            # up = 0
            #
            # for i in range(len(fitted_offer)):
            #     p_hat_gag = np.mean(fitted_offer[:i + 1])
            #     p_gag = np.mean(history[:i + 1])
            #
            #     up += ((history[i] - p_gag) * (fitted_offer[i] - p_hat_gag))
            #
            # down = 0
            # down_left = 0
            # down_right = 0
            #
            # for i in range(len(fitted_offer)):
            #     p_hat_gag = np.mean(fitted_offer[:i + 1])
            #     p_gag = np.mean(history[:i + 1])
            #     down_left += (history[i] - p_gag) ** 2
            #     down_right += (fitted_offer[i] - p_hat_gag) ** 2
            # down += np.sqrt(down_left * down_right)

            # gamma = up / down
            x = np.array(history)
            y = np.array(fitted_offer)

            if len(x) != len(y):
                x, y = fill_with_mean(x, y)

            corr = calc_corr(x, y)
            gamma = correlation_to_probability(corr)

            print(f"for cell: {index} reservation point: {self.random_reservation_points[index]} gamma: {gamma}")
            self.correlations.append(gamma)
            self.probs[index].learn(likelihood=gamma)

            index += 1
