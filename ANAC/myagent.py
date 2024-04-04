import random
from negmas.gb.common import GBState
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOResponse, SAOState


class AwesomeNegotiator(SAONegotiator):
    IP = 0  # Initial price will be set during negotiation start
    RP = 0  # Reserve price
    T = 0  # Deadline
    beta = 0  # Concession parameter

    rational_outcomes = tuple()

    partner_reserved_value = 0

    def on_negotiation_start(self, state: GBState) -> None:
        # initialize the parameters
        self.IP = 0.7
        self.RP = self.reserved_value
        self.T = self.nmi.n_steps
        self.beta = 100

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
