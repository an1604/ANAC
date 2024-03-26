"""
Submitted to ANAC 2024 Automated Negotiation League
Team: tbd
Authors: Omer (omer@bartfeld.net)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random
import math
from typing import Tuple
from scipy.optimize import curve_fit

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class ConcessionCurve:
    """
    Represents a concession curve for the opponent's strategy.
    """
    def __init__(self, max_utility: float, reserved_value: float, exponent: float):
        self.max_utility = max_utility
        self.reserved_value = reserved_value
        self.exponent = exponent

    def utility(self, time: float) -> float:
        """
        Calculate the opponent's utility at the given time.
        """
        return self.reserved_value + (self.max_utility - self.reserved_value) * (1.0 - math.pow(time, self.exponent))


class ImprovedNegotiator(SAONegotiator):
    """
    An improved negotiator agent that can learn the opponent's concession strategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concession_curve: ConcessionCurve = None
        self.offer_history: list[Tuple[float, Outcome]] = []

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.
        """
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]
        self.deadline = self.nmi.n_steps
        self.concession_curve = None
        self.offer_history = []

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.
        """
        offer = state.current_offer
        self.offer_history.append((state.relative_time, offer))
        self.update_concession_curve(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        Determine whether to accept the offer.
        """
        offer = state.current_offer

        if offer is None:
            return False

        offer_utility = self.ufun(offer)
        reserved_value = self.ufun.reserved_value

        # Calculate the maximum utility value
        max_utility = max(self.ufun(_) for _ in self.nmi.outcome_space.enumerate_or_sample())

        # Calculate the acceptance threshold based on the relative time
        relative_time = state.relative_time
        acceptance_threshold = self.ufun.reserved_value + (1.0 - relative_time) * (max_utility - self.ufun.reserved_value)

        return offer_utility >= acceptance_threshold

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Determine the counter offer.
        """
        rational_outcomes = [
            _ for _ in self.rational_outcomes if self.ufun(_) >= self.ufun.reserved_value
        ]

        if not rational_outcomes:
            return None

        # Sort the rational outcomes based on the agent's utility in descending order
        rational_outcomes.sort(key=self.ufun, reverse=True)

        # Find the outcome with the highest utility for the agent that is also
        # acceptable for the opponent (above the opponent's estimated reserved value)
        for outcome in rational_outcomes:
            if self.concession_curve and self.concession_curve.utility(state.relative_time) <= self.opponent_ufun(outcome):
                return outcome

        # If no outcome is acceptable for both parties, return the best outcome for the agent
        return rational_outcomes[0]

    def update_concession_curve(self, state: SAOState) -> None:
        """
        Update the estimated concession curve of the opponent.
        """
        if self.ufun is None or self.opponent_ufun is None or not self.offer_history:
            return

        # Calculate the opponent's utilities and relative times from the offer history
        opponent_utilities = [self.opponent_ufun(offer) for _, offer in self.offer_history]
        relative_times = [time for time, _ in self.offer_history]

        # Fit the concession curve parameters using curve_fit
        try:
            params, _ = curve_fit(
                lambda t, max_util, rv, exp: rv + (max_util - rv) * (1.0 - math.pow(t, exp)),
                relative_times, opponent_utilities,
                bounds=((0.0, 0.0, 0.2), (1.0, 1.0, 5.0))
            )
            self.concession_curve = ConcessionCurve(*params)
        except:
            pass
# Run a small tournament for testing
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(ImprovedNegotiator, small=True)
