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
        self.Nt = Nt  # Number of columns
        self.Np = Np  # Number of rows
        self.cells = []  # List to store detecting cells
        self.random_reservation_points = []  # List to store random reservation points
        self.current_time = 0  # Current negotiation time

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

    def print_detecting_region(self):
        print(f"Detecting region: {self.detecting_region}")
        for idx, cell in enumerate(self.cells):
            print(f"Cell {idx}: {cell}")
        for idx, point in enumerate(self.random_reservation_points):
            print(f"Random reservation point {idx}: {point}")

    def generate_regression_curve(self, history: List[float]):
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
                history[i] = (
                    history[i] if history[i] != init_price else init_price - 1
                )
                # print(f"------------i: {i}-----------")
                # print(f"init_price: {init_price}")
                # print(f"history[i]: {history[i]}")
                # print(f"p_i_x: {p_i_x}")
                p_star_i = np.log((init_price - history[i]) / (init_price - p_i_x))
                # print(f"p_star_i: {p_star_i}")
                t_star_i = np.log(i / t_i_x)
                # print(f"t_star_i: {t_star_i}")
                up += (p_star_i - p_i_x) * (t_star_i - t_i_x)
                # print(f"up: {up}")
                down += (t_star_i - t_i_x) ** 2
                # print(f"down: {down}")

            beta = up / down
            print(f"beta: {beta}")

        offer = init_price + (p_i_x - init_price) * (self.current_time / t_i_x) ** beta
        print(f"regression curve offer: {offer}")


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

    def on_negotiation_start(self, state: GBState) -> None:
        # initialize the parameters
        self.IP = 0.7
        self.RP = self.reserved_value
        self.T = self.nmi.n_steps
        self.beta = 100
        self.N = (3, 3)
        self.HISTORY = []
        self.detecting_region = DetectingRegion(self.T, self.N[0], self.N[1])
        # self.detecting_region.print_detecting_region()

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
        # print(f"time: {state.step}")
        # print(f"offer: {self.opponent_ufun(offer) * 100}")

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self.HISTORY.append(self.opponent_ufun(offer) * 100)

        if len(self.HISTORY) > 100:
            self.detecting_region.update_detecting_region(
                state, self.opponent_ufun(offer) * 100
            )
            # self.detecting_region.print_detecting_region()
            self.detecting_region.generate_regression_curve(self.HISTORY)

        self.update_partner_reserved_value(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun

        offer = state.current_offer

        if self.isFinalRound(state):
            # print(f"final round, accepting offer: {offer} with utility: {self.ufun(offer)}")
            return True

        if self.ufun(offer) >= self.ufun(self.bidding_strategy(state)):
            # print(f"Accepting offer: {offer} with utility: {self.ufun(offer)}")
            return True

        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        assert self.ufun

        t = state.step

        target_offer = self.IP + (self.RP - self.IP) * (t / self.T) ** self.beta
        # print(f"Target offer: {target_offer}")

        if target_offer < self.reserved_value:
            target_offer = self.reserved_value
            # print(f"Target offer: {target_offer} is below the reserved value")

        closest_outcome = None
        min_distance = float("inf")

        for outcome in self.rational_outcomes:
            outcome_utility = self.ufun(outcome)
            distance = abs(outcome_utility - target_offer)
            if distance < min_distance:
                min_distance = distance
                closest_outcome = outcome

        # print(f"Closest outcome: {closest_outcome} with utility: {self.ufun(closest_outcome)}")
        return closest_outcome

    def update_partner_reserved_value(self, state: SAOState) -> None:
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = float(self.opponent_ufun(offer)) / 2

        rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]

    # def _on_negotiation_end(self, state: GBState) -> None:
    #     print(
    #         f"Final score: {self.ufun(state.agreement) - self.reserved_value * self.opponent_ufun(state.agreement)-0.2}"
    #     )

    # def on_round_start(self, state: SAOState) -> None:
    # print(f"offer: {state.current_offer} with utility: {self.ufun(state.current_offer)} and opponent utility: {self.opponent_ufun(state.current_offer)}")

    def isFinalRound(self, state: SAOState) -> bool:
        if state.step == self.T - 1:
            return True
        return False


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
