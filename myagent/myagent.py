import math
import random

from negmas.sao import SAOResponse, SAONegotiator
from negmas import Outcome, ResponseType, SAOState, Agent, PreferencesChange
from negmas.preferences import nash_points, pareto_frontier
from negmas.preferences import nash_points, pareto_frontier


def get_random_rv(time_lower_bound, time_upper_bound, price_lower_bound, price_upper_bound):
    rand_time = random.uniform(time_lower_bound, time_upper_bound)
    rand_price = random.uniform(price_lower_bound, price_upper_bound)
    return rand_time, rand_price


class MyAgent(SAONegotiator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # The Detection region components
        self.divisions = (2, 3)
        self.self_DetReg = None
        self.self_cells = None
        self.opponent_DetReg = None
        self.opponent_cells = None
        self.first_play = True

        # General components
        self.rounds = 0
        self.self_rational_outcomes = None
        self.opponent_rational_outcomes = None
        self.deadline_time = None
        self.history = []

        # The Learning components
        self.opponent_likelihood_for_each_cell = {}

        # Two Maps between the offers and the outcomes for both agents
        self.self_offers_map = {}
        self.opponent_offers_map = {}

        # Sum of utilities for both agents for calculating some probabilities in the future.
        self.self_sum_of_utilities = 0.0
        self.opponent_sum_of_utilities = 0.0

        # Parteo frontier and Nash points of both agents based on both outcome spaces.
        self.self_Nash_equilibrium_information = None
        self.opponent_Nash_equilibrium_information = None

    def on_preferences_changed(self, changes) -> None:
        if self.ufun is None:
            return

        self.deadline_time = self.nmi.n_steps

        # Initialize the rational outcomes and sum of utilities for both agents.
        self.self_rational_outcomes = [
            _
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) > self.ufun.reserved_value
            # and self.ufun.difference_prob(first=_, second=self.equilibrium)
            # TODO: INITIALIZE THE EQUILIBRIUM
        ]
        self.self_sum_of_utilities = sum([self.ufun(_) for _ in self.self_rational_outcomes])

        self.opponent_rational_outcomes = [
            _
            for _ in self.opponent_ufun.outcome_space
            if self.opponent_ufun(_) > self.opponent_ufun(self.opponent_ufun.worst())
        ]
        self.opponent_sum_of_utilities = sum([self.opponent_ufun(_) for _ in self.opponent_rational_outcomes])

        self.initialize_detection_region(opponent_initial_value=self.opponent_ufun(
            self.opponent_ufun.best()))  # First initialization of the DetReg for both agents

        self.build_agents_maps()  # First initialization of the two offer's maps for both agents

        self.initialize_nash_information()  # Initialization of the Nash information for both agents

    def initialize_detection_region(self, low_bound_time=0, opponent_initial_value=None,
                                    divisions=(3, 4), first_play=False, op_rv=None) -> None:
        """The initialization of the detection region for both agents:
        1. The low_bound_time firstly equal 0, and every round he updated according to the relative time.
        2. The opponent_initial_value is None every single time,
            instead of the first play that carries the estimated initial value of the opponent,
            otherwise, the default value is set.
        3. The op_rv is None every single time,
             because we need to pass the opponent's estimated reservation value at each round and the historical offers.
        """
        # Opponent beliefs
        self.opponent_DetReg = {
            'time_low': low_bound_time,
            'time_high': self.deadline_time,
            'initial_value': opponent_initial_value if opponent_initial_value is not None else
            self.opponent_DetReg['initial_value'],
            'reserved_value': op_rv if op_rv else 0.0,  # TODO: CHANGE THE INITIALIZATION FROM 0!
        }

        rec = self.set_opponent_rectangle(opponent_initial_value,
                                          first_play)  # Check for the exact settings of the opponent rectangle and then call the division function.
        self.opponent_cells = self.divide_rectangle_to_cells(
            rectangle=rec, first_play=first_play,
            on_agent=False)  # Initialization of the cells for the opponent's DetReg.

        # Agent's beliefs
        self.self_DetReg = {
            'time_low': low_bound_time,
            'time_high': self.deadline_time,
            'initial_value': self.ufun.max(),
            'reserved_value': self.ufun.reserved_value
        }
        rec = (self.self_DetReg['time_low'], self.self_DetReg['time_high'], self.self_DetReg['initial_value'],
               self.self_DetReg['reserved_value'])
        self.self_cells = self.divide_rectangle_to_cells(rectangle=rec, first_play=first_play, on_agent=True)

    def divide_rectangle_to_cells(self, rectangle, first_play=False, on_agent=False) -> dict:
        T_low, T_high, P_low, P_high = rectangle
        cols, rows = self.divisions

        cell_width = (T_high - T_low) / cols
        cell_height = (P_high - P_low) / rows

        cells = []
        for i in range(cols):
            T_left = T_low + i * cell_width
            T_right = T_left + cell_width
            for j in range(rows):
                P_top = P_low + j * cell_height
                P_bottom = P_top + cell_height
                cells.append((T_left, T_right, P_top, P_bottom))

        output = {}
        for i, cell in enumerate(cells):
            output[i + 1] = {
                'cell': cell,
                'cell_index': i + 1,
            }

        if first_play:
            all_cells = cols * rows
            for k, v in output.items():
                v['likelihood'] = 1 / all_cells  # If we at the beginning of the negotiation,
                # we initialized the likelihoods using the uniform distribution.

        else:  # Otherwise,
            # we use the last values as the likelihoods before we update them (in the __call__() function below).
            for k, v in output.items():
                v['likelihood'] = self.self_cells[k]['likelihood'] if on_agent else self.opponent_cells[k]['likelihood']
        return output

    def set_opponent_rectangle(self, opponent_initial_value, first_play) -> tuple[float, float, float, float]:
        # For sure, this is the first initialization
        if first_play and opponent_initial_value is not None:
            rec = (self.opponent_DetReg['time_low'], self.opponent_DetReg['time_high'],
                   opponent_initial_value, self.opponent_DetReg['reserved_value'])

        # Otherwise, we keep rec as the values in the dictionary
        else:
            rec = (self.opponent_DetReg['time_low'], self.opponent_DetReg['time_high'],
                   self.opponent_DetReg['initial_value'], self.opponent_DetReg['reserved_value'])
        return rec

    def build_agents_maps(self) -> None:
        """Maps the agent's offers to their respective information, such as the offer's representation, ranking rate,
         lucia rate (the actual probability that the agent will accept a given offer), and the outcome itself.
         Note that this mapping is per agent, not shared map,
         all mapping focused and depend on utility function of both agents.
         """
        # TODO: ADD MORE PARAMS FOR THE MAPPINGS

        for self_offer in self.self_rational_outcomes:
            outcome = self.ufun(self_offer)
            ranking_rate = self.ufun.rank(outcomes=[self_offer])
            print(f"From build_agents_maps(), ranking_rate: {ranking_rate}")
            self.self_offers_map[self_offer] = {
                'offer': self_offer,
                'ranking_rate': ranking_rate,
                'outcome': outcome,
                'lucia_rate': outcome / self.self_sum_of_utilities
            }

        for opponent_offer in self.opponent_rational_outcomes:
            outcome = self.opponent_ufun(opponent_offer)
            ranking_rate = self.opponent_ufun.rank(outcomes=[opponent_offer])
            self.opponent_offers_map[opponent_offer] = {
                'offer': opponent_offer,
                'ranking_rate': ranking_rate,
                'outcome': outcome,
                'lucia_rate': outcome / self.opponent_sum_of_utilities
            }

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
            'nash_points': self_NP
        }
        self.opponent_Nash_equilibrium_information = {
            'parteo_frontier': opponent_pf,
            'nash_points': opponent_NP
        }

    def __call__(self, state: SAOState) -> SAOResponse:
        offer = state.current_offer
        if offer:
            self.update_general_information(offer=offer, relative_time=state.relative_time, _time=state.time)
        self.update_opponent_reserved_value(offer=offer, relative_time=state.relative_time, _time=state.time)

    def update_general_information(self, relative_time, offer, _time) -> None:
        if self.first_play:  # If we got the actual first offer from the opponent (and not None value).
            self.first_play = False
            self.initialize_detection_region(low_bound_time=relative_time,
                                             opponent_initial_value=self.opponent_ufun(offer),
                                             first_play=True)
        # Otherwise, we update the detection region as usual.
        else:
            self.initialize_detection_region(low_bound_time=relative_time)

        self_outcome = self.ufun(offer)
        opponent_outcome = self.opponent_ufun(offer)
        self.history.append({
            'offer': offer,
            'relative_time': relative_time,
            'self_outcome': self_outcome,
            'opponent_outcome': opponent_outcome,
            'round': self.rounds,
            'time': _time,
            # TODO: PRECISE THE REPRESENTATION OF THE RANK FOR BOTH AGENT AND OPPONENT!
            'self_numeric_value': self.ufun.rank(outcomes=[offer]),
            'opponent_numeric_value': self.opponent_ufun.rank(outcomes=[offer])
        })  # Append the offer to the historical offers.
        self.rounds += 1

        # TODO: I AM NOT SURE ABOUT THIS ADDITION FOR SUN OF UTILITIES, CHECK THIS OUT!
        self.self_sum_of_utilities += self_outcome
        self.opponent_sum_of_utilities += opponent_outcome

    def update_opponent_reserved_value(self, offer, relative_time, _time):
        random_rv_from_each_cell = self.generate_random_reservation_points()

        cells_regression_curves = self.generate_regression_curves(cells_rv=random_rv_from_each_cell, _time=_time,
                                                                  relative_time=relative_time)

        # Generated the fitted offers for each cell
        fitted_offers = self.generate_fitted_offers(cells_regression_curves=cells_regression_curves)

        # Calculating the non-linear correlation coefficient between the real and the fitted offers for each cell
        non_linear_correlation_coefficient = self.calc_non_linear_correlation_coefficient(fitted_offers, _time,
                                                                                          relative_time)

    def generate_random_reservation_points(self, on_agent=False) -> list[tuple[float, tuple[float, float]]]:
        """Generate a single random reservation point for each cell according to on_agent condition.
        Return:
            reservation points in a list, where each element is a tuple of the cell index (first element)
            and the random reservation point itself (second element) represented by tuple too.
        """
        res = []  # List of tuples: (cell_index, rand_rv)
        if not on_agent:
            for opponent_index, opponent_cell in self.opponent_cells.items():
                t_low, t_high, p_low, p_high = opponent_cell['cell']  # Extract the boundaries from the tuple
                rand_rv = get_random_rv(time_lower_bound=t_low, time_upper_bound=t_high,
                                        price_lower_bound=p_low, price_upper_bound=p_high)
                res.append(
                    (opponent_index, rand_rv)
                )
        else:
            for self_index, self_cell in self.opponent_cells.items():
                t_low, t_high, p_low, p_high = self_cell['cell']
                rand_rv = get_random_rv(time_lower_bound=t_low, time_upper_bound=t_high,
                                        price_lower_bound=p_low, price_upper_bound=p_high)
                res.append(
                    (self_index, rand_rv)
                )
        return res

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
            p0 = self.opponent_DetReg['initial_price']
            for cell_index, cell_rv in cells_rv:
                tix, pix = cell_rv
                b = self.calc_regression_coefficient(p0, tix, pix, current_round, _time)

                # TODO: TRY TO USE RELATIVE_TIME INSTEAD OF _TIME!
                reg_curve = p0 + ((pix - p0) * pow((_time / tix), b))

                curves[cell_index] = {
                    'cell': self.opponent_cells[cell_index]['cell'],
                    'cell_index': cell_index,
                    'regression_curve': (_time, reg_curve)  # TODO: CHECK HOW TO EXACTLY REPRESENT THE REGRESSION CURVE!
                }
        else:
            p0 = self.self_DetReg['initial_price']
            for cell_index, cell_rv in cells_rv:
                tix, pix = cell_rv
                b = self.calc_regression_coefficient(p0, tix, pix, current_round, _time)
                reg_curve = p0 + ((pix - p0) * pow((_time / tix), b))
                curves[cell_index] = {
                    'cell': self.self_cells[cell_index]['cell'],
                    'cell_index': cell_index,
                    'regression_curve': (_time, reg_curve)
                }
        return curves

    def calc_regression_coefficient(self, p0, tix, pix, current_round, _time) -> float:
        # Keep the historical offers that in the time boundaries
        offers_in_time_boundaries = [offer for offer in self.history if
                                     offer['round'] <= current_round and offer['time'] <= _time]
        numerator = 0.0
        denominator = 0.0
        for offer in offers_in_time_boundaries:
            a = (p0 - offer['price']) / (p0 - pix)
            p_star = math.log(a)
            c = _time / tix  # TODO: CHECK IF THIS EQUATION IS GOOD OR NEED TO REPLACE _TIME WITH CURRENT_ROUND!
            t_star = math.log(c)
            numerator += p_star * t_star
            denominator += pow(t_star, 2)
        return numerator / denominator

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
        for cell_index, cell_info in cells_regression_curves.items():
            t, y = cell_info['regression_curve']
            fitted_offers[cell_index] = []
            for offer in self.history:
                # TODO: trying to calculate x according to the time AND relative time!
                x = offer['time']
                x_1 = offer['relative_time']

                # we place x in the linear equation according to the regression curve params (t,y)
                y_hat = (t * x) + y
                y_hat1 = (t * x_1) + y

                fitted_offer = (x, y_hat)  # The result of the linear equation represented by 2D point.
                fitted_offer_1 = (x_1, y_hat1)

                fitted_offers[cell_index].append(fitted_offer)
                fitted_offers[cell_index].append(fitted_offer_1)

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
        for cell_index, fitted_offers_list in fitted_offers.items():
            for offer in fitted_offers_list:
                t, p = offer
                average_of_fitted_offers += p
                num_of_fitted_offers += 1
        average_of_fitted_offers /= num_of_fitted_offers

        # Calculating the average price for all historical offers.
        average_of_historical_offers = 0.0
        num_of_offers = 0
        for offer in self.history:
            p_opp = offer['opponent_outcome']
            # p_self = offer['self_outcome']

            average_of_historical_offers += p_opp
            # TODO: CHECK -> average_of_historical_offers += p_self
            num_of_offers += 1
        average_of_historical_offers /= num_of_offers

        coeffs = {}
        for cell_index, fitted_offers_list in fitted_offers.items():
            numerator = 0.0
            denominator = 0.0
            denominator_second_part = 0.
            denominator_first_part = 0.
            for historical_offer in self.history:
                p_opp = historical_offer['opponent_outcome']
                p_self = historical_offer['self_outcome']

                # Separating the denominator to two prats,
                # because the first part handles only the historical offer and the second handles only
                # the fitted offers, and the length not necessarily equals.
                denominator_first_part += pow((p_opp - average_of_historical_offers), 2)

                for fitted_offer in fitted_offers_list:
                    t_fitted, p_fitted = fitted_offer
                    numerator += (p_opp - average_of_historical_offers) * (p_fitted - average_of_fitted_offers)
                    denominator_second_part += pow((p_fitted - average_of_fitted_offers), 2)
            # Combining the two parts by multiplication.
            denominator = denominator_second_part * denominator_first_part
            coeffs[cell_index] = numerator / denominator
        return coeffs
