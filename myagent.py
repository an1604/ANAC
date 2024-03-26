"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* Adaptive Bayesian Negotiators
*Authors* John Doe (john.doe@example.com), Jane Smith (jane.smith@example.com)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
from scipy.stats import norm
import random
from typing import Optional

import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class AdaptiveBayesianNegotiator(SAONegotiator):
    """
    A negotiator that uses Bayesian learning to estimate the opponent's reservation value
    and incorporates opponent modeling and dynamic bidding strategies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rational_outcomes = tuple()
        self.flag = False
        self.past_offers = []

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.
        """
        if self.ufun is None:
            return
        if not self.flag:
            self.prior_rv = self.ufun.reserved_value  # Initial guess for opponent's reservation value
            self.posterior_rv = self.prior_rv  # Current estimate of opponent's reservation value
            self.opponent_behavior = None  # Estimate of opponent's behavior (e.g., conceding, boulware, etc.)
            self.bidding_strategy = self.conservative_bidding  # Initial bidding strategy
            self.prior_mean = self.ufun.reserved_value  # Prior mean for opponent's reservation value
            self.prior_variance = 1.0  # Prior variance for opponent's reservation value
            self.flag = True
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
        self.update_opponent_behavior(state)
        self.update_bidding_strategy(state)
        self.update_past_offers(offer)

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

    def conservative_bidding(self, state: SAOState) -> Outcome:
        """
        Conservative bidding strategy that offers the outcome that maximizes our utility
        while being rational for the opponent.
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

    def conceding_bidding(self, state: SAOState) -> Outcome:
        """
        Conceding bidding strategy that offers outcomes closer to the opponent's estimated
        reservation value as time passes.
        """
        rational_outcomes_for_opponent = [
            outcome
            for outcome in self.rational_outcomes
            if self.opponent_ufun(outcome) > self.posterior_rv
        ]

        if not rational_outcomes_for_opponent:
            # If no rational outcomes for the opponent, offer our best outcome
            return self.ufun.best()

        # Sort the rational outcomes by the opponent's utility in descending order
        rational_outcomes_for_opponent.sort(key=self.opponent_ufun, reverse=True)

        # Calculate concession rate based on opponent behavior
        if self.opponent_behavior == "conceding":
            concession_rate = 1.0 - state.relative_time
        else:
            concession_rate = 0.5

        # Offer an outcome with concession rate applied
        cutoff = int(len(rational_outcomes_for_opponent) * concession_rate)
        return random.choice(rational_outcomes_for_opponent[cutoff:])

    def update_past_offers(self, offer: Outcome) -> None:
        """
        Update the list of past offers.
        """
        if offer is not None:
            self.past_offers.append(offer)
    def update_partner_reserved_value(self, state: SAOState) -> None:
        """
        Update the estimated opponent's reservation value using Bayesian learning.
        """
        offer = state.current_offer

        if offer is None:
            return

        opponent_utility = self.opponent_ufun(offer)

        # Update the posterior distribution of the opponent's reservation value
        self.posterior_rv, self.posterior_variance = self.update_posterior(self.prior_mean, self.prior_variance, opponent_utility)

        # Update the list of rational outcomes based on the new estimate
        self.rational_outcomes = [
            outcome
            for outcome in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(outcome) > self.ufun.reserved_value
            and self.opponent_ufun(outcome) > self.posterior_rv
        ]

    def update_posterior(self, prior_mean: float, prior_variance: float, opponent_utility: float) -> (float, float):
        """
        Update the posterior distribution of the opponent's reservation value using Bayesian inference.
        """
        # Calculate likelihood of observed opponent utility given the opponent's reservation value
        likelihood = norm.pdf(opponent_utility, prior_mean, np.sqrt(prior_variance))

        # Update posterior parameters using Bayesian inference
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0)
        posterior_mean = posterior_variance * (prior_mean / prior_variance + opponent_utility / prior_variance)

        return posterior_mean, posterior_variance

    def update_opponent_behavior(self, state: SAOState) -> None:
        """
        Update the estimate of the opponent's behavior based on historical negotiation data.
        """
        if len(self.past_offers) == 0:
            return

        # Calculate average utility of past offers
        avg_utility = np.mean([self.opponent_ufun(offer) for offer in self.past_offers])

        # Update opponent behavior based on average utility
        if avg_utility < self.ufun.reserved_value:
            self.opponent_behavior = "conservative"
        else:
            self.opponent_behavior = "conceding"

    def update_bidding_strategy(self, state: SAOState) -> None:
        """
        Update the bidding strategy based on the estimated opponent's behavior.
        """
        if self.opponent_behavior == "conceding":
            self.bidding_strategy = self.conceding_bidding
        else:
            self.bidding_strategy = self.conservative_bidding

    # ... (rest of the code remains the same) ...
# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AdaptiveBayesianNegotiator, small=True)