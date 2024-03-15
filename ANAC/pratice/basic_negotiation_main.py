import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from anl.anl2024 import anl2024_tournament
from anl.anl2024.negotiators import Boulware, Conceder, RVFitter
from anl.anl2024.negotiators import Boulware
from negmas.sao import SAOResponse, SAONegotiator
from negmas import Outcome, ResponseType, SAOState

# Imports for running a single negotiation
import copy
from negmas.sao import SAOMechanism
from anl.anl2024.runner import mixed_scenarios
from anl.anl2024.negotiators.builtins import Linear


def aspiration_function(t, mx, rv, e):
    return (mx - rv) * (1.0 - np.power(t, e)) + rv


class MyNegotiator(SAONegotiator):
    def __init__(self, *args, e: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e = e
        self.opponent_times: list[float] = []  # Keeps track the opponent offers (by times)
        self.opponents_utilities: list[float] = []  # Keeps track of opponent utilities of it's offers
        self._past_opponent_rv = 0.0  # Keeps track of our last estimate of the opponent reserved value
        # Keeps track the rational outcome set given our estimate of the
        # opponent reserved value and our knowledge of ours
        self._rational: list[tuple[float, float, Outcome]] = []

    def __call__(self, state: SAOState):
        self.update_reserved_value(state.current_offer, state.relative_time)
        # Run the acceptance strategy, and if the offer received is acceptable, accept it
        if self.is_acceptable(state.current_offer, state.relative_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)
        return SAOResponse(ResponseType.REJECT_OFFER, self.generate_offer(state.relative_time))

    def generate_offer(self, relative_time) -> Outcome:
        # The offering strategy
        if not self._rational \
                or abs(self.opponent_ufun.reserved_value - self._past_opponent_rv) > 1e-3:
            # We try to find the best offer for me and for the opponent as well
            self._rational = sorted(
                [
                    (my_util, opp_util, _)
                    for _ in self.nmi.outcome_space.enumerate_or_sample(
                    levels=10, max_cardinality=100_00
                )
                    if (my_util := float(self.ufun(_))) > self.ufun.reserved_value
                       and (opp_util := float(self.opponent_ufun(_))) > self.opponent_ufun.reserved_value
                ]
            )
        # If there are no rational outcomes (e.g., our estimate of the opponent rv is very wrong)
        # then just revert to offering our top offer
        if not self._rational:
            return self.ufun.best()
        asp = aspiration_function(relative_time, 1.0, 0.0, self.e)
        max_rational = len(self._rational) - 1
        idx = max(0, min(max_rational, int(asp * max_rational)))
        outcome = self._rational[idx][-1]
        return outcome

    def is_acceptable(self, offer, relative_time) -> bool:
        if offer is None:  # If the offer is None, we are just at the beginning of a new negotiation
            return False
        asp = aspiration_function(
            relative_time, 1.0, self.ufun.reserved_value, self.e
        )
        return float(self.ufun(offer)) >= asp

    def update_reserved_value(self, offer, relative_time):
        """Learns the reserved value of the partner"""
        if offer is None:
            return
        # Save the current received from the opponent and their times
        self.opponents_utilities.append(float(self.opponent_ufun(offer)))
        self.opponent_times.append(relative_time)

        bounds = ((0.2, 0.0), (5.0, min(self.opponents_utilities)))
        try:
            optimal_vals, _ = curve_fit(
                lambda x, e, rv: aspiration_function(
                    x, self.opponents_utilities[0], rv, e
                ),
                self.opponent_times,
                self.opponents_utilities,
                bounds=bounds,
            )
            self._past_opponent_rv = self.opponent_ufun.reserved_value
            self.opponent_ufun.reserved_value = optimal_vals[1]
        except Exception as e:
            pass


if __name__ == '__main__':
    # Evaluate the negotiators
    res = anl2024_tournament(
        n_scenarios=1, n_repetitions=3, nologs=True, njobs=-1,
        competitors=[MyNegotiator, Boulware, Conceder]
    )

    # Running a single negotiation
    s = mixed_scenarios(1)[0]  # Get some random scenario
    # Copy ufuns and set the reserved value to 0 at the beginning
    ufuns0 = [copy.deepcopy(u) for u in s.ufuns]
    for u in ufuns0:
        u.reserved_value = 0.0
    # The negotiation mechanism
    session = SAOMechanism(n_steps=1000, outcome_space=s.outcome_space)
    # Add the negotiators into the session
    session.add(
        MyNegotiator(name="MyNegotiator",
                     private_info=dict(opponent_ufun=ufuns0[1]),
                     ufun=ufuns0[0])
    )
    session.add(RVFitter(name="RVFitter",
                         private_info=dict(opponent_ufun=ufuns0[0]),
                         ufun=s.ufuns[1]))

    session.run()
    session.plot()
    plt.show()
