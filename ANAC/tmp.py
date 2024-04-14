"""
**Submitted to ANAC 2024 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2024 ANL competition.
"""
import random

from negmas.gb.common import GBState
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState
from negmas.preferences import nash_points, pareto_frontier
import matplotlib.pyplot as plt
import numpy as np

class AwesomeNegotiator(SAONegotiator):
    """
    Your agent code. This is the ONLY class you need to implement
    """
    op_offer_history = []
    me_history = []
    rational_outcomes = tuple()
    threshold = 0

    partner_reserved_value = 0

    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2024, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """
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
        if offer is not None:
            self.op_offer_history.append(offer)
        # Determine the acceptability of the offer in the acceptance_strategy
        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        # If it's not acceptable, determine the counter offer in the bidding_strategy
        return SAOResponse(ResponseType.REJECT_OFFER, self.bidding_strategy(state))
    def on_negotiation_start(self, state: GBState) -> None:
        """
        Called when the negotiation starts.

        Args:
            state: the `GBState` containing the negotiation information (e.g. the negotiation ID, the role, etc).

        Remarks:
            - Can optionally be used for initializing your agent.
        """
        self.threshold = 100
    
    
    def scoreFun(self,offer):
        s = (self.ufun(offer) - self.reserved_value)*(self.opponent_ufun(offer) - self.partner_reserved_value)
        print(f'SCORE: {s}')
        return s
    
    def acceptance_strategy(self, state: SAOState) -> bool:
        """
        This is one of the functions you need to implement.
        It should determine whether or not to accept the offer.

        Returns: a bool.
        """
        assert self.ufun

        offer = state.current_offer
        
        if self.scoreFun(offer) > self.threshold:
            return True
        return False

    def bidding_strategy(self, state: SAOState) -> Outcome | None:
        """
        This is one of the functions you need to implement.
        It should determine the counter offer.

        Returns: The counter offer as Outcome.
        """
        # print('ASdsadasdsadasdasdasd')
        
        a = self.opponent_ufun(state.current_offer)
        for fuckingshit in self.rational_outcomes:
            # print(f'fuckingshit : {fuckingshit}')
            if a == self.ufun(fuckingshit):
                return fuckingshit
            
        # The opponent's ufun can be accessed using self.opponent_ufun, which is not used yet.

        return random.choice(self.rational_outcomes)

    def update_partner_reserved_value(self, state: SAOState) -> None:
        """This is one of the functions you can implement.
        Using the information of the new offers, you can update the estimated reservation value of the opponent.

        returns: None.
        """
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer

        if self.opponent_ufun(offer) < self.partner_reserved_value:
            self.partner_reserved_value = self.opponent_ufun(offer)
            self.partner_reserved_value = -1 * (self.scoreFun(offer) / (self.ufun(offer)- self.reserved_value )) + self.opponent_ufun(offer)
        print(f'op RV: {self.partner_reserved_value}')

        # update rational_outcomes by removing the outcomes that are below the reservation value of the opponent
        # Watch out: if the reserved value decreases, this will not add any outcomes.
        rational_outcomes = self.rational_outcomes = [
            _
            for _ in self.rational_outcomes
            if self.opponent_ufun(_) > self.partner_reserved_value
        ]


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(AwesomeNegotiator, small=True)
