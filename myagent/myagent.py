import math
import random
from scipy.linalg._matfuncs import eps
from negmas.sao import SAOResponse, SAONegotiator
from negmas import Outcome, ResponseType, SAOState
from negmas.preferences import nash_points, pareto_frontier
from negmas.preferences import nash_points, pareto_frontier, winwin_level
from detection_region import DetectionRegionHandler
from helpers_functions import nash_optimality, sigmoid, custom_max_offers, print_situation
from ga import solve
import numpy as np
from offer import Offer, Historical_Offer, Offer_Map


class MyAgent(SAONegotiator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # The Detection region components
        self.divisions = (2, 3)
        self.self_DetReg = None
        self.opponent_DetReg = None
        self.first_play = True
        self.estimated_rv = None
        self.estimated_ip = None

        # General components
        self.self_offer_name = 'self_offer'
        self.opponent_offer_name = 'opponent_offer'
        self.rounds = 0
        self._eps = eps
        self.self_rational_outcomes = None
        self.opponent_rational_outcomes = None
        self.deadline_time = None
        self.history = []
        self.self_rv_offers = None

        # Two Maps between the offers and the outcomes for both agents
        self.self_offers_map = Offer_Map()
        self.opponent_offers_map = Offer_Map()

        # Sum of utilities for both agents for calculating some probabilities in the future.
        self.self_sum_of_utilities = 0.0
        self.opponent_sum_of_utilities = 0.0

        # Parteo frontier and Nash points of both agents based on both outcome spaces.
        self.self_Nash_equilibrium_information = None
        self.opponent_Nash_equilibrium_information = None
        self.nash_optimality_values = []

    def on_preferences_changed(self, changes) -> None:
        if self.ufun is None:
            return

        self.deadline_time = self.nmi.n_steps

        # Initialize the rational outcomes and sum of utilities for both agents.
        self.self_rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) >= self.ufun.reserved_value
            # and self.ufun.difference_prob(first=_, second=self.equilibrium)
            # TODO: INITIALIZE THE EQUILIBRIUM
        ]

        self.self_sum_of_utilities = sum([self.ufun(_) for _ in self.self_rational_outcomes])

        opponent_lower_bound_outcome = self.opponent_ufun(self.opponent_ufun.worst()) if self.opponent_ufun(
            self.opponent_ufun.worst()) > 0 else 0.15

        self.opponent_rational_outcomes = [
            _
            for _ in self.opponent_ufun.outcome_space
            if self.opponent_ufun(_) > opponent_lower_bound_outcome
        ]
        self.opponent_sum_of_utilities = sum([self.opponent_ufun(_) for _ in self.opponent_rational_outcomes])

        self.build_agents_maps()  # First initialization of the two offer's maps for both agents

        self.estimated_rv = self.opponent_offers_map.get_lowest()
        self.estimated_ip = self.opponent_offers_map.get_highest()

        self.opponent_DetReg = DetectionRegionHandler(divisions=self.divisions, deadline_time=self.deadline_time,
                                                      initial_value=self.estimated_ip.ranking_rate,
                                                      time_low_bound=0.,
                                                      reserved_value=self.estimated_rv.ranking_rate,
                                                      first_play=self.first_play
                                                      )
        self.self_DetReg = DetectionRegionHandler(divisions=self.divisions,
                                                  deadline_time=self.deadline_time,
                                                  initial_value=self.self_offers_map.get_highest().ranking_rate,
                                                  time_low_bound=0.,
                                                  reserved_value=self.self_offers_map.get_lowest().ranking_rate,
                                                  first_play=self.first_play)

        self.initialize_nash_information()  # Initialization of the Nash information for both agents

    def build_agents_maps(self) -> None:
        """Maps the agent's offers to their respective information, such as the offer's representation, ranking rate,
         lucia rate (the actual probability that the agent will accept a given offer), and the outcome itself.
         Note that this mapping is per agent, not shared map,
         all mapping focused and depend on utility function of both agents.
         """

        for self_offer in self.self_rational_outcomes:
            outcome = self.ufun(self_offer)
            ranking_rate = self.ufun.rank_with_weights(outcomes=[self_offer])[0][1]

            self.self_offers_map.add_offer(Offer(
                name=self.self_offer_name,
                offer=self_offer,
                ranking_rate=ranking_rate,
                outcome=outcome,
                lucia_rate=outcome / self.self_sum_of_utilities
            ))

        for opponent_offer in self.opponent_rational_outcomes:
            outcome = self.opponent_ufun(opponent_offer)
            ranking_rate = self.opponent_ufun.rank_with_weights(outcomes=[opponent_offer])[0][1]
            self.opponent_offers_map.add_offer(Offer(
                name=self.opponent_offer_name,
                offer=opponent_offer,
                ranking_rate=ranking_rate,
                outcome=outcome,
                lucia_rate=outcome / self.opponent_sum_of_utilities
            ))

    def initialize_nash_information(self) -> None:
        self_pf = pareto_frontier(ufuns=[self.ufun, self.opponent_ufun],
                                  outcomes=self.self_rational_outcomes,
                                  issues=self.nmi.issues)

        opponent_pf = pareto_frontier(ufuns=[self.opponent_ufun, self.ufun],
                                      outcomes=self.opponent_rational_outcomes,
                                      issues=self.nmi.issues)

        self_NP = nash_points(ufuns=[self.ufun, self.opponent_ufun],
                              outcome_space=self.nmi.outcome_space,
                              outcomes=self.self_rational_outcomes,
                              issues=self.nmi.issues,
                              frontier=list(self_pf)[0])

        opponent_NP = nash_points(ufuns=[self.ufun, self.opponent_ufun],
                                  outcome_space=self.opponent_ufun.outcome_space,
                                  outcomes=self.opponent_rational_outcomes,
                                  issues=self.nmi.issues,
                                  frontier=list(opponent_pf)[0])

        # Initialize the dictionaries with the information gathered.
        self.self_Nash_equilibrium_information = {
            'parteo_frontier': self_pf,
            'nash_points': [
                (NP[0], self.self_rational_outcomes[NP[1]])
                for NP in self_NP
            ]
        }
        self.opponent_Nash_equilibrium_information = {
            'parteo_frontier': opponent_pf,
            'nash_points': [(NP[0], self.opponent_rational_outcomes[NP[1]])
                            for NP in opponent_NP
                            ]
        }

    def __call__(self, state: SAOState) -> SAOResponse:
        offer = state.current_offer
        if offer:
            situation = {
                'offer': offer if offer is not None else 'None',
                'current_time': state.time,
                'relative time': state.relative_time,
                'round': self.rounds,
                'offers history': self.history
            }
            print_situation(situation)

            self.update_general_information(offer=offer, relative_time=state.relative_time, _time=state.time)

            self.update_opponent_reserved_value(offer=offer, relative_time=state.relative_time, _time=state.time)

            if self.is_accepted(offer=offer, _time=state.time, relative_time=state.relative_time):
                print(
                    f"Successfully accepted {offer}!\nOur utility: {self.ufun(offer)}\n opponent utility: {self.opponent_ufun(offer)}")
                return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

            bid = self.generate_offer(offer=offer, _time=state.time, relative_time=state.relative_time)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        return SAOResponse(ResponseType.REJECT_OFFER, self.ufun.best())

    def update_general_information(self, relative_time, offer, _time) -> None:
        self.rounds += 1
        self.opponent_DetReg.set_lower_bound_time(lower_bound_time=relative_time)
        self.self_DetReg.set_lower_bound_time(lower_bound_time=relative_time)

        self_outcome = self.ufun(offer)
        opponent_outcome = self.opponent_ufun(offer)
        self.self_sum_of_utilities += self_outcome
        self.opponent_sum_of_utilities += opponent_outcome

        self_ranking_rate = self.ufun.rank_with_weights(outcomes=[offer])[0][1]
        self_lu_rate = self_outcome / self.self_sum_of_utilities

        opponent_ranking_rate = self.opponent_ufun.rank_with_weights(outcomes=[offer])[0][1]
        opponent_lu_rate = opponent_outcome / self.opponent_sum_of_utilities
        self.history.append(
            Historical_Offer(offer=offer,
                             current_time=_time,
                             relative_time=relative_time,
                             current_round=self.rounds,
                             self_outcome=self_outcome,
                             opponent_outcome=opponent_outcome,
                             self_lucia_rate=self_lu_rate,
                             opponent_lucia_rate=opponent_lu_rate,
                             self_ranking_rate=self_ranking_rate,
                             opponent_ranking_rate=opponent_ranking_rate
                             ))

        if self.first_play:  # If we got the actual first offer from the opponent (and not None value).
            self.first_play = False
            # Check if the opponent really give me his best offer,
            # and if he does, we initialize the initial price again with this price (normalized!)
            if self.estimated_ip.outcome - self._eps <= opponent_ranking_rate <= self.estimated_ip.outcome + self._eps:
                self.estimated_ip = Offer(
                    name='opponent_estimated_initial_value',
                    offer=offer,
                    ranking_rate=opponent_ranking_rate,
                    lucia_rate=opponent_lu_rate,
                    outcome=opponent_outcome
                )

        # Updating the maps for both agents according to the offer occurred.
        if not self.self_offers_map.exists(offer=offer):
            s_offer = Offer(
                name=self.self_offer_name,
                offer=offer,
                ranking_rate=self_ranking_rate,
                outcome=self_outcome,
                lucia_rate=self_lu_rate
            )
            self.self_offers_map.add_offer(s_offer)

        if not self.opponent_offers_map.exists(offer=offer):
            o_offer = Offer(
                name=self.opponent_offer_name,
                offer=offer,
                ranking_rate=opponent_ranking_rate,
                outcome=opponent_outcome,
                lucia_rate=opponent_lu_rate
            )
            self.opponent_offers_map.add_offer(o_offer)

    def update_opponent_reserved_value(self, offer, relative_time, _time):
        if _time == 0.:
            _time += random.uniform(0, 0.001)

        random_rv_from_each_cell = self.opponent_DetReg.generate_random_reservation_points()

        cells_regression_curves = self.generate_regression_curves(cells_rv=random_rv_from_each_cell, _time=_time,
                                                                  relative_time=relative_time)

        # Generated the fitted offers for each cell
        fitted_offers = self.generate_fitted_offers(cells_regression_curves=cells_regression_curves)

        # Calculating the non-linear correlation coefficient between the real and the fitted offers for each cell
        # In addition,
        # these coefficients represent the likelihoods
        # that the reserved value of the opponent will be in cell i.
        non_linear_correlation_coefficient = self.calc_non_linear_correlation_coefficient(fitted_offers, _time,
                                                                                          relative_time)
        self.opponent_DetReg.update_probabilities(non_linear_correlation_coefficient=non_linear_correlation_coefficient,
                                                  _round=self.rounds)

    def calc_regression_coefficient(self, p0, tix, pix, current_round, _time) -> float:
        # Keep the historical offers that in the time boundaries
        numerator = 0.0
        denominator = 0.0
        for offer in self.history:
            # print(f"Historical offer: {offer.__dict__}")
            _, offer = self.self_offers_map.get_offer_from_tuple(
                offer=offer.offer)  # This offer.offer is from type Historical_Offer,
            # we make him type Offer.

            a = (p0 - offer.ranking_rate) / (p0 - pix)
            print(f'a before changing: {a}')
            if a <= 0:  # If the value of `a` is less o equal to zero, we update him to a very close to 0 value,
                # to affect the value of the log to be very close to negative inf.
                a = random.uniform(0, 0.01)
            print(f'a after changing: {a}')
            p_star = math.log(a)

            c = _time / tix  # TODO: CHECK IF THIS EQUATION IS GOOD OR NEED TO REPLACE _TIME WITH CURRENT_ROUND!

            print(f'c before changing: {c}')
            if c <= 0:
                c = random.uniform(0, 0.01)
            print(f'c after changing: {c}')

            t_star = math.log(c)
            numerator += p_star * t_star
            denominator += pow(t_star, 2)
        print(f"numerator:{numerator}, denominator{denominator}")
        return numerator / denominator

    def generate_regression_curves(self, cells_rv, _time, relative_time, on_agent=False) -> dict:
        """
        Generates the regression curves for each cell based on the current time.
        For now, we see the regression line as a linear function as the following:
                                    y = ax + b
        Where:
            ax is the current time in the negotiation multiplied by x.
            b is the regression coefficient for the current time in the negotiation.
        """
        curves = {}
        current_round = self.rounds
        if not on_agent:
            p0 = self.opponent_DetReg._DetReg['initial_value']
            for cell_key, cell_rv in cells_rv:
                tix, pix = cell_rv
                b = self.calc_regression_coefficient(p0, tix, pix, current_round, _time)

                # TODO: TRY TO USE RELATIVE_TIME INSTEAD OF _TIME!
                reg_curve = p0 + ((pix - p0) * pow((_time / tix), b))

                curves[cell_key] = {
                    'cell': self.opponent_DetReg._cells[cell_key]['cell'],
                    'cell_index': cell_key,
                    'regression_curve': (_time, reg_curve)  # TODO: CHECK HOW TO EXACTLY REPRESENT THE REGRESSION CURVE!
                }
        else:
            p0 = self.self_DetReg['initial_value']
            for cell_key, cell_rv in cells_rv:
                tix, pix = cell_rv
                b = self.calc_regression_coefficient(p0, tix, pix, current_round, _time)
                reg_curve = p0 + ((pix - p0) * pow((_time / tix), b))
                curves[cell_key] = {
                    'cell': self.self_DetReg._cells[cell_key]['cell'],
                    'cell_index': cell_key,
                    'regression_curve': (_time, reg_curve)
                }
        return curves

    def generate_fitted_offers(self, cells_regression_curves) -> dict[int, list]:
        """
        For generating the fitted offers,
        we're using the linear equation for each regression curve that we're calculating in the previous round.
        According to the following formula:
                    y^(x) = tx + y
        Where:
            tx is the current time in the negotiation multiplied by the time that the offer occurs (from the history of the previous round).
            y is the current regression coefficient that we found.
        Args:
            cells_regression_curves:
            dictionary of all the regression curves generated by the previous round,accompanied by generated random offers.

        Returns:
            dictionary of all the fitted offers for each cell.
        """
        fitted_offers = {}
        for cell_key, cell_info in cells_regression_curves.items():
            t, y = cell_info['regression_curve']
            fitted_offers[cell_key] = []
            for offer in self.history:
                # TODO: trying to calculate x according to the time AND relative time!
                x = offer.current_time
                x_1 = offer.relative_time

                # we place x in the linear equation according to the regression curve params (t,y)
                y_hat = (t * x) + y
                y_hat1 = (t * x_1) + y

                fitted_offer = (x, y_hat)  # The result of the linear equation represented by 2D point.
                fitted_offer_1 = (x_1, y_hat1)

                fitted_offers[cell_key].append(fitted_offer)
                fitted_offers[cell_key].append(fitted_offer_1)

        return fitted_offers

    def calc_non_linear_correlation_coefficient(self, fitted_offers, _time, relative_time) -> dict[int, float]:
        """
        Calculates the non-linear correlation coefficient between the given fitted offers and the historical offers for each cell.
        Actually,
        the non-linear correlation coefficient is the prior probability
        that the reserved value of the opponent will be in cell i.
        Args:
            fitted_offers: dictionary of all the fitted offers for each cell.
            _time: the current time in the negotiation.
            relative_time: the time relative to the deadline time of the negotiation.

        Returns:
            dictionary of all the non-linear correlation coefficients for each cell.

        Note:
            The non-linear correlation (in range [0,1]) is a parameter reflecting the non-linear similarity between
            the fitted offers and the historical offers, this is an important parameter to be used in Bayesian learning for
            the belief updating.
        """
        # Calculating the average price for all the fitted offers till time t.
        average_of_fitted_offers = 0.0
        num_of_fitted_offers = 0
        for cell_key, fitted_offers_list in fitted_offers.items():
            for offer in fitted_offers_list:
                t, p = offer
                average_of_fitted_offers += p
                num_of_fitted_offers += 1
        average_of_fitted_offers /= num_of_fitted_offers

        # Calculating the average price for all historical offers.
        average_of_historical_offers = 0.0
        num_of_offers = 0
        for offer in self.history:
            p_opp = offer.opponent_outcome
            # p_self = offer['self_outcome']

            average_of_historical_offers += p_opp
            # TODO: CHECK -> average_of_historical_offers += p_self
            num_of_offers += 1
        average_of_historical_offers /= num_of_offers

        coeffs = {}
        for cell_key, fitted_offers_list in fitted_offers.items():
            numerator = 0.0
            denominator = 0.0

            if len(self.history) > 0:
                denominator_second_part = 0.
                denominator_first_part = 0.
            else:
                denominator_second_part = random.uniform(0.0001, 0.001)
                denominator_first_part = random.uniform(0.0001, 0.001)

            for historical_offer in self.history:
                p_opp = historical_offer.opponent_outcome
                p_self = historical_offer.self_outcome

                # Separating the denominator to two prats,
                # because the first part handles only the historical offer and the second handles only
                # the fitted offers, and the length not necessarily equals.
                denominator_first_part += pow((p_opp - average_of_historical_offers), 2)

                for fitted_offer in fitted_offers_list:
                    t_fitted, p_fitted = fitted_offer
                    numerator += (p_opp - average_of_historical_offers) * (p_fitted - average_of_fitted_offers)
                    denominator_second_part += pow((p_fitted - average_of_fitted_offers), 2)
            # Combining the two parts by multiplication.
            if denominator_second_part == 0.:
                denominator_second_part = random.uniform(0.0001, 0.001)
            if denominator_first_part == 0.:
                denominator_first_part = random.uniform(0.0001, 0.001)
            print(
                f"denominator_second_part: {denominator_second_part}, denominator_first_part: {denominator_first_part} "
            )
            denominator = denominator_second_part * denominator_first_part
            coeffs[cell_key] = numerator / denominator
        return coeffs

    def is_accepted(self, offer, _time, relative_time, a=10., b=0.5,
                    THRESHOLD=1.):
        if offer is None:
            return False

        # best_estimation_cell = self.opponent_DetReg.get_best_cell(_round=self.rounds)
        self_offer_idx, self_offer = self.self_offers_map.get_offer_from_tuple(offer)
        opponent_offer_idx, opponent_offer = self.opponent_offers_map.get_offer_from_tuple(offer)

        if self_offer >= self.ufun.reserved_value:
            if self_offer.ranking_rate >= opponent_offer.ranking_rate:
                return True
            if self_offer.lucia_rate <= opponent_offer.lucia_rate:
                return True
            z = a * (self_offer.outcome - self.ufun.reserved_value) + b * self_offer.ranking_rate
            p = sigmoid(z)
            return p < self_offer.ranking_rate

        # Check the nash optimally value passing some THRESHOLD (out target is to maximize this value)
        self.estimated_rv = self.opponent_DetReg.get_estimated_rv(_round=self.rounds)
        self.nash_optimality_values.append(nash_optimality(utility1=self_offer.outcome, rv1=self.ufun.reserved_value,
                                                           utility2=opponent_offer.outcome, rv2=self.estimated_rv))
        if self.nash_optimality_values[-1] > THRESHOLD:
            return True

        # Check if the offer in the nash points (in the range aspect).
        for NP in self.self_Nash_equilibrium_information['nash_points']:
            np_range, _ = NP
            rng = range(np_range[0], np_range[1])
            if self_offer.outcome in rng:
                return True

        for NP in self.opponent_Nash_equilibrium_information['nash_points']:
            np_range, _ = NP
            rng = range(np_range[0], np_range[1])
            if self_offer.outcome in rng:
                return True
        return False

    def generate_offer(self, offer, _time, relative_time):
        # If the offer is None, we generate some random nash offer to the opponent
        if offer is None:
            return random.choice(
                [self.self_rational_outcomes[NP[1]]
                 for NP in self.self_Nash_equilibrium_information['nash_points']]
            )
        else:
            self_best_offer = self.self_offers_map.sort(reverse=True)[0]
            opponent_best_offer = self.opponent_offers_map.sort(reverse=True)[0]
            ga_offer = solve(points=[self_best_offer, opponent_best_offer])
            return ga_offer


if __name__ == "__main__":
    from helpers.runner import run_a_tournament

    run_a_tournament(MyAgent, small=True, nologs=True)
