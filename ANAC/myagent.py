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

        self.initialize_detecting_region()

    def initialize_detecting_region(self):
        # Define detecting region
        self.detecting_region = (0, self.T, 0, 1)  # Assuming the opponent's IP and RP are in the range [0, 1]

        # Initialize detecting cells
        for t_idx in range(self.Nt):
            t_low = self.detecting_region[0] + t_idx * (self.detecting_region[1] - self.detecting_region[0]) / self.Nt
            t_high = self.detecting_region[0] + (t_idx + 1) * (self.detecting_region[1] - self.detecting_region[0]) / self.Nt
            for p_idx in range(self.Np):
                p_low = self.detecting_region[2] + p_idx * (self.detecting_region[3] - self.detecting_region[2]) / self.Np
                p_high = self.detecting_region[2] + (p_idx + 1) * (self.detecting_region[3] - self.detecting_region[2]) / self.Np
                rv = random.uniform(p_low, p_high)  # Random reserved value in each cell
                self.cells.append(((t_low, t_high, p_low, p_high), rv))

    def get_cell_reserved_value(self, cell_idx: int) -> float:
        return self.cells[cell_idx][1]
    
    def select_random_reservation_points(self):
        for idx, ((t_low, t_high, p_low, p_high), _) in enumerate(self.cells):
            t_x_i = random.uniform(t_low, t_high)
            p_x_i = random.uniform(p_low, p_high)
            self.cells[idx] = ((t_low, t_high, p_low, p_high), (t_x_i, p_x_i))

    def calculate_regression_lines(self, historical_offers: List[float]):
        for idx, ((t_low, t_high, p_low, p_high), (t_x_i, p_x_i)) in enumerate(self.cells):
            t_values = np.linspace(t_low, t_high, 10)
            regression_values = []
            for t_val in t_values:
                relevant_offers = [offer for offer in historical_offers if offer[0] <= t_val]
                if relevant_offers:
                    t_vals = [offer[0] for offer in relevant_offers]
                    p_vals = [offer[1] for offer in relevant_offers]
                    t_star = np.log(t_vals / t_x_i)
                    p_star = np.log((p_x_i - p_vals[0]) / (p_x_i - p_low))
                    b = np.sum(t_star * p_star) / np.sum(t_star ** 2)
                    p_star_values = p_star[0] + (p_star[0] - np.log(p_x_i)) * (t_star / np.log(t_x_i))
                    p_values = np.exp(p_star_values)
                    regression_values.append(np.mean(p_values))
                else:
                    regression_values.append(p_low)
            self.cells[idx] = ((t_low, t_high, p_low, p_high), (t_x_i, p_x_i), t_values, regression_values)

    def calculate_fitted_offers(self):
        fitted_offers = []
        for idx, (_, _, t_values, regression_values) in enumerate(self.cells):
            for t_val, p_val in zip(t_values, regression_values):
                fitted_offers.append((t_val, p_val))
        print(fitted_offers)
        return fitted_offers


class AwesomeNegotiator(SAONegotiator):
    IP = 0  # Initial price will be set during negotiation start
    RP = 0  # Reserve price
    T = 0  # Deadline
    beta = 0  # Concession parameter
    detecting_region: DetectingRegion
    N = (0,0) # (rows, columns) to divide the deterministic region into N 
    HISTORY = [] # History of the offers made by the opponent
    
    
    rational_outcomes = tuple()
    partner_reserved_value = 0


    def on_negotiation_start(self, state: GBState) -> None:
        # initialize the parameters
        self.IP = 0.7
        self.RP = self.reserved_value
        self.T = self.nmi.n_steps
        self.beta = 100
        self.N = (4, 4)
        self.HISTORY = []
        self.detecting_region = DetectingRegion(self.T, 3, 4)  # Initialize with 3 columns and 4 rows
        
        # Step 1: Select random reservation points
        self.detecting_region.select_random_reservation_points()

        # Step 2: Calculate regression lines using all historical offers
        self.detecting_region.calculate_regression_lines(self.HISTORY)

        # Step 3: Calculate fitted offers
        self.fitted_offers = self.detecting_region.calculate_fitted_offers()
        

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
        
        # Update the history of the offers made by the opponent
        if offer is not None:
            self.HISTORY.append(self.opponent_ufun(offer))        

        self.update_partner_reserved_value(state)

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

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
