"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* tbd
*Authors* Omer (omer@bartfeld.net)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random
import math

from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class ImprovedNegotiator(SAONegotiator):
    """
    An improved negotiator agent
    """

    rational_outcomes = tuple()
    partner_reserved_value = 0

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

        self.partner_reserved_value = self.ufun.reserved_value

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

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
        acceptance_threshold = self.ufun.reserved_value + (1.0 - relative_time) * (
                    max_utility - self.ufun.reserved_value)

        return offer_utility >= acceptance_threshold

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        Determine the counter offer.
        """
        offer = state.current_offer
        partner_reserved_value = self.partner_reserved_value

        # Filter out outcomes that are below the agent's reserved value
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
            if self.opponent_ufun(outcome) > partner_reserved_value:
                return outcome

        # If no outcome is acceptable for both parties, return the best outcome for the agent
        return rational_outcomes[0]

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Update the estimated reservation value of the opponent.
        """
        if self.ufun is None or self.opponent_ufun is None:
            return

        offer = state.current_offer

        if offer is None:
            return

        partner_utility = self.opponent_ufun(offer)

        if partner_utility < self.partner_reserved_value:
            self.partner_reserved_value = partner_utility


# Run a small tournament for testing
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(ImprovedNegotiator, small=True)