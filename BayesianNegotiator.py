"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Bayesian Learners
*Authors* John Doe (john.doe@example.com), Jane Smith (jane.smith@example.com)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random
from typing import Optional

import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class BayesianNegotiator(SAONegotiator):
    """
    A negotiator that uses Bayesian learning to estimate the opponent's reservation value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rational_outcomes = tuple()



    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.
        """
        if self.ufun is None:
            return
        self.prior_rv = self.ufun.reserved_value  # Initial guess for opponent's reservation value
        self.posterior_rv = self.prior_rv  # Current estimate of opponent's reservation value
        self.rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
        ]

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called to (counter-)offer.
        """
        offer = state.current_offer

        self.update_partner_reserved_value(state)

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        if self.acceptance_strategy(state, offer):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))

    def acceptance_strategy(self, state: SAOState, offer: Optional[Outcome]) -> bool:
        """
        Acceptance strategy based on the estimated opponent's reservation value.
        """
        if offer is None:
            return False

        # Accept the offer if it is better than the estimated opponent's reservation value
        return self.ufun(offer) > self.posterior_rv

    def bidding_strategy(self, state: SAOState) -> Outcome:
        """
        Bidding strategy based on the estimated opponent's reservation value.
        """
        rational_outcomes_for_opponent = [
            outcome
            for outcome in self.rational_outcomes
            if self.opponent_ufun(outcome) > self.posterior_rv
        ]

        if not rational_outcomes_for_opponent:
            # If no rational outcomes for the opponent, offer our best outcome
            return self.ufun.best()

        # Offer the outcome that maximizes our utility while being rational for the opponent
        return max(rational_outcomes_for_opponent, key=self.ufun)

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Update the estimated opponent's reservation value using Bayesian learning.
        """
        offer = state.current_offer

        if offer is None:
            return

        opponent_utility = self.opponent_ufun(offer)

        # Update the posterior distribution of the opponent's reservation value
        self.posterior_rv = self.update_posterior(self.prior_rv, opponent_utility)

        # Update the list of rational outcomes based on the new estimate
        self.rational_outcomes = [
            outcome
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(outcome) > self.ufun.reserved_value
            and self.opponent_ufun(outcome) > self.posterior_rv
        ]

    def update_posterior(self, prior_rv: float, opponent_utility: float) -> float:
        """
        Update the posterior distribution of the opponent's reservation value using Bayesian learning.
        """
        # Implement your Bayesian learning algorithm here
        # This is a simple example that updates the posterior based on the maximum likelihood estimate
        if opponent_utility < prior_rv:
            return opponent_utility
        else:
            return prior_rv

    # ... (rest of the code remains the same) ...
# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(BayesianNegotiator, small=True)