import random
from typing import List
from negmas.gb.common import GBState
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
import math
import numpy as np


class DetectingRegion:
    def __init__(self, T: int, Nt: int, Np: int):
        self.T = T  # Deadline
        self.current_time = 0  # Current negotiation time
        self.Nt = Nt  # Number of columns
        self.Np = Np  # Number of rows
        self.cells = []  # List to store detecting cells
        self.random_reservation_points = []  # List to store random reservation points
        self.regression_curves = []  # List to store regression curves
        self.fitted_offers = []  # List to store fitted offers
        self.correlations = []  # List to store non-linear correlations
        self.prior_probabilities = []  # List to store prior probabilities
        self.posterior_probabilities = []  # List to store posterior probabilities
        self.fitted_offers_T = []  # List to store fitted offers till deadline

        self.initialize_detecting_region()

    def initialize_detecting_region(self):
        # Define detecting region
        self.detecting_region = (1, self.T, 0, 100)  # Initial detecting region

        # Initialize detecting cells and generate random reservation points
        self.update_detecting_region()

    def update_detecting_region(self, state: SAOState = None, max_price=100):
        # Update the detecting region with the current negotiation time
        if state is not None:
            self.current_time = state.step
        else:
            self.current_time = 0
        max_price = max_price
        if self.current_time == 0:
            max_price = 100
        self.detecting_region = (self.T, self.T, 0, max_price)

        # Clear previous random reservation points and detecting cells
        self.random_reservation_points.clear()
        self.cells.clear()
        self.regression_curves.clear()
        self.fitted_offers.clear()
        self.correlations.clear()
        self.prior_probabilities.clear()
        self.posterior_probabilities.clear()

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
                px = random.uniform(p_low, p_high)
                self.cells.append((t_low, t_high, p_low, p_high))
                self.random_reservation_points.append((self.T, px))
                self.prior_probabilities.append(1 / (self.Nt * self.Np))
                self.posterior_probabilities.append(1 / (self.Nt * self.Np))

    def print_detecting_region(self):
        print(f"Detecting region: {self.detecting_region}")
        for idx, cell in enumerate(self.cells):
            print(f"Cell {idx}: {cell}")
        for idx, point in enumerate(self.random_reservation_points):
            print(f"Random reservation point {idx}: {point}")

    def generate_regression_curve(self, history: List[float]):
        # init price
        init_price = history[0] if history[0] != 0 else 100

        for reservation_point in self.random_reservation_points:
            # claculate the beta coefficient
            beta = 0
            up = 0
            down = 0
            t_i_x, p_i_x = reservation_point[0], reservation_point[1]

            for i in range(1, self.current_time):
                history[i] = history[i] if history[i] != init_price else init_price - 1
                p_star_i = np.log((init_price - history[i]) / (init_price - p_i_x))
                t_star_i = np.log(i / self.T)
                up += p_star_i * t_star_i
                down += t_star_i**2

            beta = up / down
            # print(f"beta: {beta}")
            if beta < 0 or np.isnan(beta):
                beta = 0.8
            self.regression_curves.append((init_price, p_i_x, t_i_x, beta))

    def clalculate_fitted_offers(self):
        for curve in self.regression_curves:
            index = 0
            offer_list = []  # List to store fitted offers
            init_price, p_i_x, t_i_x, beta = curve
            for i in range(0, self.current_time + 1):
                offer = init_price + (p_i_x - init_price) * ((i / t_i_x) ** beta)
                offer_list.append(offer)
            index += 1
            self.fitted_offers.append(offer_list)

    def clalculate_fitted_offers_till_deadtime(self):
        self.fitted_offers_T.clear()
        for curve in self.regression_curves:
            offer_list = []  # List to store fitted offers
            init_price, p_i_x, t_i_x, beta = curve
            for i in range(0, self.T + 1):
                offer = init_price + (p_i_x - init_price) * ((i / t_i_x) ** beta)
                offer_list.append((i, offer))
            self.fitted_offers_T.append(offer_list)

    def get_non_linear_correlation(self, history: List[float]):
        p_gag = np.mean(history)
        for fitted_offer in self.fitted_offers:
            p_hat_gag = np.mean(fitted_offer)
            up = 0
            down = 0
            down_left = 0
            down_right = 0
            for i in range(0, self.current_time):
                left = history[i] - p_gag
                right = fitted_offer[i] - p_hat_gag
                up += left * right
                down_left += left**2
                down_right += right**2
            down = math.sqrt(down_left * down_right)
            correlation = up / down
            if correlation < 0 or np.isnan(correlation):
                correlation = 0.1
            self.correlations.append(correlation)

    def bayesian_update(self):
        p = 0
        sum = 0

        for idx, cell in enumerate(self.cells):
            # print(f"Prior probability for cell {idx}: {self.prior_probabilities[idx]}")
            # print(f"Correlation for cell {idx}: {self.correlations[idx]}")
            sum += self.prior_probabilities[idx] * self.correlations[idx]
            # print(f"Sum: {sum}")

        for idx, cell in enumerate(self.cells):
            p = (self.prior_probabilities[idx] * self.correlations[idx]) / sum
            # print(
            # f"Posterior probability for reservation point {self.random_reservation_points[idx]}: {p / sum}"
            # )
            self.posterior_probabilities[idx] = p
            self.prior_probabilities[idx] = p

    def get_best_reservation_point(self):
        best_point = self.random_reservation_points[0]
        best_probability = self.posterior_probabilities[0]

        for idx, point in enumerate(self.random_reservation_points):
            if self.posterior_probabilities[idx] > best_probability:
                best_probability = self.posterior_probabilities[idx]
                best_point = point

        # print(f"Best reservation point: {best_point} with probability: {best_probability}")
        return best_point


class AwesomeNegotiator(SAONegotiator):
    IP = 0  # Initial price will be set during negotiation start
    RP = 0  # Reserve price
    T = 0  # Deadline
    beta = 0  # Concession parameter
    detecting_region: DetectingRegion
    N = (0, 0)  # (rows, columns) to divide the deterministic region into N
    HISTORY = []  # History of the offers made by the opponent
    Historical_offer_points = []  # each point is a tuple of (time, utility)

    rational_outcomes = tuple()
    partner_reserved_value = 0
    my_current_offer = (0, 0)
    my_reserved_price = 0
    MY_HISTORY = []  # History of the offers made by the agent
    score = 0
    score_list = []

    def on_negotiation_start(self, state: GBState) -> None:
        # initialize the parameters
        self.IP = 101 - self.ufun(self.ufun.best()) * 100
        self.RP = 100 - self.reserved_value * 100
        self.T = self.nmi.n_steps
        self.beta = 1.6
        self.N = (3, 3)
        self.HISTORY = []
        self.detecting_region = DetectingRegion(self.T, self.N[0], self.N[1])
        self.my_reserved_price = 100 - self.reserved_value * 100
        self.my_current_offer = (0, self.IP)

    def on_preferences_changed(self, changes):
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is not None:
            self.HISTORY.append(self.opponent_ufun(offer) * 100)

        if len(self.HISTORY) > 4:
            self.detecting_region.update_detecting_region(
                state, self.opponent_ufun(offer) * 100
            )
            self.detecting_region.generate_regression_curve(self.HISTORY)
            self.detecting_region.clalculate_fitted_offers()
            self.detecting_region.get_non_linear_correlation(self.HISTORY)
            self.detecting_region.bayesian_update()

            if state.step > 2:
                self.adapt_new_beta(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        offer = state.current_offer

        if self.isFinalRound(state):
            return True

        if self.ufun(offer) >= self.ufun(self.bidding_strategy(state)):
            return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        assert self.ufun

        t = state.step
        t0 = self.my_current_offer[0]
        p0 = self.my_current_offer[1]

        target_offer = p0 + (self.my_reserved_price - p0) * (
            ((t - t0) / (self.T - t0)) ** self.beta
        )

        if target_offer > self.my_reserved_price:
            target_offer = self.my_reserved_price / 2

        closest_outcome = None
        target_utility = 1 - target_offer / 100
        min_distance = float("inf")

        for outcome in self.rational_outcomes:
            distance = abs(target_utility - self.ufun(outcome))
            if distance < min_distance:
                min_distance = distance
                closest_outcome = outcome

        self.my_current_offer = (state.step, 100 - self.ufun(closest_outcome) * 100)
        self.MY_HISTORY.append(self.my_current_offer)

        return closest_outcome

    def isFinalRound(self, state: SAOState) -> bool:
        if state.step == int(self.T * 0.8):
            return True
        return False

    def adapt_new_beta(self, state: SAOState) -> None:

        self.detecting_region.clalculate_fitted_offers_till_deadtime()
        beta_gags = []  # List to store beta values
        conssesion_points = []
        p0 = self.my_current_offer[1]
        t0 = self.my_current_offer[0]

        nego_region = [(100 - 100 * self.ufun(_)) for _ in self.rational_outcomes]
        for idx, point in enumerate(self.detecting_region.random_reservation_points):
            pix = 100 - point[1]
            tix = point[0] - 1
            if (pix) > p0 and pix < self.my_reserved_price:
                conssesion_points.append((tix, pix))
            else:
                offers = [
                    fitted_offer
                    for fitted_offer in self.detecting_region.fitted_offers_T[idx]
                    if (100 - fitted_offer[1]) > min(nego_region)
                    and (100 - fitted_offer[1]) < max(nego_region)
                    and fitted_offer[0] > t0
                ]
                if len(offers) > 0:
                    conssesion_points.append(min(offers, key=lambda x: x[1]))
                else:
                    conssesion_points.append((t0 + 1, 0.97 * self.my_reserved_price))

        if len(conssesion_points) > 0:
            for point in conssesion_points:
                tp = point[0]
                pp = point[1]
                log_base = (tp - t0) / (self.T - t0)
                log_body = (p0 - pp) / (p0 - self.my_reserved_price)

                if log_base != 1 and log_body != 1 and log_base > 0 and log_body > 0:
                    new_beta = math.log(log_body) / math.log(log_base)
                    beta_gags.append(new_beta)
        down = 0
        for beta, prior in zip(beta_gags, self.detecting_region.prior_probabilities):
            down += prior / (1 + beta)
        if down == 0:
            self.beta = 0.8
            return
        overall_beta = (1 / down) - 1
        if overall_beta < 0 or np.isnan(overall_beta):
            overall_beta = 0.8
        self.beta = overall_beta


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
import random
from typing import List
from negmas.gb.common import GBState
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
import math
import numpy as np


class DetectingRegion:
    def __init__(self, T: int, Nt: int, Np: int):
        self.T = T  # Deadline
        self.current_time = 0  # Current negotiation time
        self.Nt = Nt  # Number of columns
        self.Np = Np  # Number of rows
        self.cells = []  # List to store detecting cells
        self.random_reservation_points = []  # List to store random reservation points
        self.regression_curves = []  # List to store regression curves
        self.fitted_offers = []  # List to store fitted offers
        self.correlations = []  # List to store non-linear correlations
        self.prior_probabilities = []  # List to store prior probabilities
        self.posterior_probabilities = []  # List to store posterior probabilities
        self.fitted_offers_T = []  # List to store fitted offers till deadline

        self.initialize_detecting_region()

    def initialize_detecting_region(self):
        # Define detecting region
        self.detecting_region = (1, self.T, 0, 100)  # Initial detecting region

        # Initialize detecting cells and generate random reservation points
        self.update_detecting_region()

    def update_detecting_region(self, state: SAOState = None, max_price=100):
        # Update the detecting region with the current negotiation time
        if state is not None:
            self.current_time = state.step
        else:
            self.current_time = 0
        max_price = max_price
        if self.current_time == 0:
            max_price = 100
        self.detecting_region = (self.T, self.T, 0, max_price)

        # Clear previous random reservation points and detecting cells
        self.random_reservation_points.clear()
        self.cells.clear()
        self.regression_curves.clear()
        self.fitted_offers.clear()
        self.correlations.clear()
        self.prior_probabilities.clear()
        self.posterior_probabilities.clear()

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
                px = random.uniform(p_low, p_high)
                self.cells.append((t_low, t_high, p_low, p_high))
                self.random_reservation_points.append((self.T, px))
                self.prior_probabilities.append(1 / (self.Nt * self.Np))
                self.posterior_probabilities.append(1 / (self.Nt * self.Np))

    def print_detecting_region(self):
        print(f"Detecting region: {self.detecting_region}")
        for idx, cell in enumerate(self.cells):
            print(f"Cell {idx}: {cell}")
        for idx, point in enumerate(self.random_reservation_points):
            print(f"Random reservation point {idx}: {point}")

    def generate_regression_curve(self, history: List[float]):
        # init price
        init_price = history[0] if history[0] != 0 else 100

        for reservation_point in self.random_reservation_points:
            # claculate the beta coefficient
            beta = 0
            up = 0
            down = 0
            t_i_x, p_i_x = reservation_point[0], reservation_point[1]

            for i in range(1, self.current_time):
                history[i] = history[i] if history[i] != init_price else init_price - 1
                p_star_i = np.log((init_price - history[i]) / (init_price - p_i_x))
                t_star_i = np.log(i / self.T)
                up += p_star_i * t_star_i
                down += t_star_i**2

            beta = up / down
            # print(f"beta: {beta}")
            if beta < 0 or np.isnan(beta):
                beta = 0.8
            self.regression_curves.append((init_price, p_i_x, t_i_x, beta))

    def clalculate_fitted_offers(self):
        for curve in self.regression_curves:
            index = 0
            offer_list = []  # List to store fitted offers
            init_price, p_i_x, t_i_x, beta = curve
            for i in range(0, self.current_time + 1):
                offer = init_price + (p_i_x - init_price) * ((i / t_i_x) ** beta)
                offer_list.append(offer)
            index += 1
            self.fitted_offers.append(offer_list)

    def clalculate_fitted_offers_till_deadtime(self):
        self.fitted_offers_T.clear()
        for curve in self.regression_curves:
            offer_list = []  # List to store fitted offers
            init_price, p_i_x, t_i_x, beta = curve
            for i in range(0, self.T + 1):
                offer = init_price + (p_i_x - init_price) * ((i / t_i_x) ** beta)
                offer_list.append((i, offer))
            self.fitted_offers_T.append(offer_list)

    def get_non_linear_correlation(self, history: List[float]):
        p_gag = np.mean(history)
        for fitted_offer in self.fitted_offers:
            p_hat_gag = np.mean(fitted_offer)
            up = 0
            down = 0
            down_left = 0
            down_right = 0
            for i in range(0, self.current_time):
                left = history[i] - p_gag
                right = fitted_offer[i] - p_hat_gag
                up += left * right
                down_left += left**2
                down_right += right**2
            down = math.sqrt(down_left * down_right)
            correlation = up / down
            if correlation < 0 or np.isnan(correlation):
                correlation = 0.1
            self.correlations.append(correlation)

    def bayesian_update(self):
        p = 0
        sum = 0

        for idx, cell in enumerate(self.cells):
            # print(f"Prior probability for cell {idx}: {self.prior_probabilities[idx]}")
            # print(f"Correlation for cell {idx}: {self.correlations[idx]}")
            sum += self.prior_probabilities[idx] * self.correlations[idx]
            # print(f"Sum: {sum}")

        for idx, cell in enumerate(self.cells):
            p = (self.prior_probabilities[idx] * self.correlations[idx]) / sum
            # print(
            # f"Posterior probability for reservation point {self.random_reservation_points[idx]}: {p / sum}"
            # )
            self.posterior_probabilities[idx] = p
            self.prior_probabilities[idx] = p

    def get_best_reservation_point(self):
        best_point = self.random_reservation_points[0]
        best_probability = self.posterior_probabilities[0]

        for idx, point in enumerate(self.random_reservation_points):
            if self.posterior_probabilities[idx] > best_probability:
                best_probability = self.posterior_probabilities[idx]
                best_point = point

        # print(f"Best reservation point: {best_point} with probability: {best_probability}")
        return best_point


class AwesomeNegotiator(SAONegotiator):
    IP = 0  # Initial price will be set during negotiation start
    RP = 0  # Reserve price
    T = 0  # Deadline
    beta = 0  # Concession parameter
    detecting_region: DetectingRegion
    N = (0, 0)  # (rows, columns) to divide the deterministic region into N
    HISTORY = []  # History of the offers made by the opponent
    Historical_offer_points = []  # each point is a tuple of (time, utility)

    rational_outcomes = tuple()
    partner_reserved_value = 0
    my_current_offer = (0, 0)
    my_reserved_price = 0
    MY_HISTORY = []  # History of the offers made by the agent
    score = 0
    score_list = []

    def on_negotiation_start(self, state: GBState) -> None:
        # initialize the parameters
        self.IP = 101 - self.ufun(self.ufun.best()) * 100
        self.RP = 100 - self.reserved_value * 100
        self.T = self.nmi.n_steps
        self.beta = 1.6
        self.N = (3, 3)
        self.HISTORY = []
        self.detecting_region = DetectingRegion(self.T, self.N[0], self.N[1])
        self.my_reserved_price = 100 - self.reserved_value * 100
        self.my_current_offer = (0, self.IP)

    def on_preferences_changed(self, changes):
        # If there a no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Estimate the reservation value, as a first guess, the opponent has the same reserved_value as you
        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if offer is not None:
            self.HISTORY.append(self.opponent_ufun(offer) * 100)

        if len(self.HISTORY) > 4:
            self.detecting_region.update_detecting_region(
                state, self.opponent_ufun(offer) * 100
            )
            self.detecting_region.generate_regression_curve(self.HISTORY)
            self.detecting_region.clalculate_fitted_offers()
            self.detecting_region.get_non_linear_correlation(self.HISTORY)
            self.detecting_region.bayesian_update()

            if state.step > 2:
                self.adapt_new_beta(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        offer = state.current_offer

        if self.isFinalRound(state):
            return True

        if self.ufun(offer) >= self.ufun(self.bidding_strategy(state)):
            return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        assert self.ufun

        t = state.step
        t0 = self.my_current_offer[0]
        p0 = self.my_current_offer[1]

        target_offer = p0 + (self.my_reserved_price - p0) * (
            ((t - t0) / (self.T - t0)) ** self.beta
        )

        if target_offer > self.my_reserved_price:
            target_offer = self.my_reserved_price / 2

        closest_outcome = None
        target_utility = 1 - target_offer / 100
        min_distance = float("inf")

        for outcome in self.rational_outcomes:
            distance = abs(target_utility - self.ufun(outcome))
            if distance < min_distance:
                min_distance = distance
                closest_outcome = outcome

        self.my_current_offer = (state.step, 100 - self.ufun(closest_outcome) * 100)
        self.MY_HISTORY.append(self.my_current_offer)

        return closest_outcome

    def isFinalRound(self, state: SAOState) -> bool:
        if state.step == int(self.T * 0.8):
            return True
        return False

    def adapt_new_beta(self, state: SAOState) -> None:

        self.detecting_region.clalculate_fitted_offers_till_deadtime()
        beta_gags = []  # List to store beta values
        conssesion_points = []
        p0 = self.my_current_offer[1]
        t0 = self.my_current_offer[0]

        nego_region = [(100 - 100 * self.ufun(_)) for _ in self.rational_outcomes]
        for idx, point in enumerate(self.detecting_region.random_reservation_points):
            pix = 100 - point[1]
            tix = point[0] - 1
            if (pix) > p0 and pix < self.my_reserved_price:
                conssesion_points.append((tix, pix))
            else:
                offers = [
                    fitted_offer
                    for fitted_offer in self.detecting_region.fitted_offers_T[idx]
                    if (100 - fitted_offer[1]) > min(nego_region)
                    and (100 - fitted_offer[1]) < max(nego_region)
                    and fitted_offer[0] > t0
                ]
                if len(offers) > 0:
                    conssesion_points.append(min(offers, key=lambda x: x[1]))
                else:
                    conssesion_points.append((t0 + 1, 0.97 * self.my_reserved_price))

        if len(conssesion_points) > 0:
            for point in conssesion_points:
                tp = point[0]
                pp = point[1]
                log_base = (tp - t0) / (self.T - t0)
                log_body = (p0 - pp) / (p0 - self.my_reserved_price)

                if log_base != 1 and log_body != 1 and log_base > 0 and log_body > 0:
                    new_beta = math.log(log_body) / math.log(log_base)
                    beta_gags.append(new_beta)
        down = 0
        for beta, prior in zip(beta_gags, self.detecting_region.prior_probabilities):
            down += prior / (1 + beta)
        if down == 0:
            self.beta = 0.8
            return
        overall_beta = (1 / down) - 1
        if overall_beta < 0 or np.isnan(overall_beta):
            overall_beta = 0.8
        self.beta = overall_beta


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
