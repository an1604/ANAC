"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Bayesian Learners
*Authors* John Doe (john.doe@example.com), Jane Smith (jane.smith@example.com)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random
from typing import List

import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class AwesomeNegotiator(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """

    rational_outcomes: List[Outcome] = []
    partner_reserved_value_samples: List[float] = []

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()  # enumerates outcome space when finite, samples when infinite
            if self.ufun(_) > self.ufun.reserved_value
        ]

        # Initialize the partner's reserved value samples with a uniform distribution
        self.partner_reserved_value_samples = np.random.uniform(0, 1, size=1000)

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.

        Args:
            state: the `SAOState` containing the offer from your partner (None if you are just starting the negotiation)
                   and other information about the negotiation (e.g. current step, relative time, etc).
        Returns:
            A response of type `SAOResponse` which indicates whether you accept, or reject the offer or leave the negotiation.
            If you reject an offer, you are required to pass a counter offer.

        Remarks:
            - This is the ONLY function you need to implement.
            - You can access your ufun using `self.ufun`.
            - You can access the opponent's ufun using self.opponent_ufun(offer)
            - You can access the mechanism for helpful functions like sampling from the outcome space using `self.nmi` (returns an `SAONMI` instance).
            - You can access the current offer (from your partner) as `state.current_offer`.
              - If this is `None`, you are starting the negotiation now (no offers yet).
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)

        # if there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer

        if self.ufun(offer) > (2 * self.ufun.reserved_value):
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        # Calculate the estimated partner's reserved value
        partner_reserved_value = np.mean(self.partner_reserved_value_samples)

        # Filter the rational outcomes based on the estimated partner's reserved value
        filtered_rational_outcomes = [
            outcome
            for outcome in self.rational_outcomes
            if self.opponent_ufun(outcome) > partner_reserved_value
        ]

        # If there are no filtered rational outcomes, return a random outcome
        if not filtered_rational_outcomes:
            return random.choice(self.rational_outcomes)

        # Otherwise, return the outcome that maximizes the product of utilities
        max_product = 0
        best_outcome = None
        for outcome in filtered_rational_outcomes:
            product = self.ufun(outcome) * (self.opponent_ufun(outcome) - partner_reserved_value)
            if product > max_product:
                max_product = product
                best_outcome = outcome

        return best_outcome

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if offer is not None:
            # Update the partner's reserved value samples based on the opponent's offer
            opponent_utility = self.opponent_ufun(offer)
            self.partner_reserved_value_samples = [
                sample for sample in self.partner_reserved_value_samples if sample < opponent_utility
            ]

            # If there are no samples left, reset with a uniform distribution
            if not self.partner_reserved_value_samples:
                self.partner_reserved_value_samples = np.random.uniform(0, 1, size=1000)


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)