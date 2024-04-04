import numpy as np
from scipy.linalg._matfuncs import eps
from helpers_functions import get_random_rv, custom_max


def _range(start, stop, step):
    while start < stop:
        yield round(start, 10)
        start += step


class DetectionRegionHandler:
    def __init__(self, divisions, deadline_time, initial_value, time_low_bound,
                 first_play=False, reserved_value=None):
        self._eps = eps
        self.deadline_time = deadline_time
        self._divisions = divisions
        self._all_cells = divisions[0] * divisions[1]
        self._first_play = first_play
        self._DetReg = {
            'time_low': time_low_bound,
            'time_high': 1.,
            'initial_value': initial_value,
            'reserved_value': reserved_value if reserved_value is not None else 0.,
        }

        self._cells = self.initialize_cells_division()

        self.probabilities_for_each_cell = {}
        for cell_key, cell_info in self._cells.items():
            self.probabilities_for_each_cell[cell_key] = {
                'prior_probability': 1 / self._all_cells
            }  # Initialize probabilities as uniform distribution for now at the beginning.

        self.best_probabilities = {}  # Keep track the best probability in each round

    def set_first_play(self, first_play):
        self._first_play = first_play

    def set_divisions(self, divisions):
        self._divisions = divisions

    def set_lower_bound_time(self, lower_bound_time):
        self._DetReg['time_low'] = float(lower_bound_time / self.deadline_time)  # Normalization
        self.update()

    def set_reserved_value(self, reserved_value):
        self._DetReg['reserved_value'] = reserved_value
        self.update()

    def set_initial_value(self, initial_value):
        self._DetReg['initial_value'] = initial_value
        self.update()

    def update(self):
        """
        We use the update function after every round in the negotiation, and update the prior probabilities based
        under two conditions:
             1. If there is a new cell generated, we calculate the prior probability based on the method of complements.
             2. If there is an existing cell generated, we give her the previous prior probability from last round.
        """
        cells = self.initialize_cells_division()  # Get the new division according to the new bounds
        probabilities_for_each_cell = {}

        # Calculate the sum of probabilities first,
        # to measure the prior probability of the other cells that are not
        # in the previous rounds of the negotiation.
        sum_prior_probs = 0.
        for cell_key, cell_info in cells.items():
            is_cell_in_cells, _, _ = self.cell_in_cells(cell_key, cell_info)
            if is_cell_in_cells and sum_prior_probs < 1.:
                sum_prior_probs += self.probabilities_for_each_cell[cell_key].values()

        for cell_key, cell_info in cells.items():
            # If the exact same cell were in the last round, we keep the prob as the same as before
            is_cell_in_cells, c_key, c_info = self.cell_in_cells(cell_key, cell_info)
            if is_cell_in_cells:
                probabilities_for_each_cell[cell_key]['prior_probability'] = c_info['prior_probability']
            # Otherwise, we use the method of complements for the prior probability
            else:
                probabilities_for_each_cell[cell_key] = {'prior_probability': 1 - (
                        sum_prior_probs / self._all_cells) if sum_prior_probs > 0 \
                    else self.calc_estimated_prior_probability_for_cell(cell_range=cell_key,
                                                                        cell_info=cell_info,
                                                                        cell=cell_info['cell'],
                                                                        all_probs=sum_prior_probs)
                                                         }

        self.probabilities_for_each_cell = probabilities_for_each_cell
        self._cells = cells

    def cell_in_cells(self, cell_key, cell_info):
        cell_lower_bound, cell_upper_bound = cell_key
        for c_key, c_info in self._cells.items():
            low, high = c_key
            rng = _range(low, high + self._eps, 0.1)  # TODO: Adjust step size as needed
            if cell_lower_bound in rng and cell_upper_bound in rng:
                return True, c_key, cell_info
        return False, None, None

    def get_detection_region(self):
        return self._DetReg

    def get_best_cell(self, _round=None):
        if _round is None:
            return self.best_probabilities
        return self.best_probabilities[_round]

    def get_estimated_rv(self, _round):
        best_cell = self.get_best_cell(_round)
        cell_key, cell_info = best_cell
        cell = cell_info['cell']
        t_low, t_high, p_low, p_high = cell

        return (t_low + t_high) / 2., (p_low + p_high) / 2.

    def initialize_cells_division(self):
        T_low, T_high, P_low, P_high = self._DetReg['time_low'], self._DetReg['time_high'], self._DetReg[
            'initial_value'], self._DetReg['reserved_value']
        cols, rows = self._divisions
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
            t_low, t_high, p_low, p_high = cell
            cell_key = tuple(
                [t_low, p_high]
            )  # We initialize the key as the biggest range in the detection region for keeping the uniqueness of each cell.
            c = {
                'cell': cell,
                'cell_index': i + 1,
                'cell_range': cell_key,
            }

            output[cell_key] = c
            # print(f"New cell generated: {c}")
        return output

    def get_prior_probabilities(self, cells_dict):
        # we use the last values as the prior_probabilities before we update them.
        for k, v in cells_dict.items():
            all_probs = sum([v['prior_probability'] for k, v in self._cells.items()])
            if k not in self._cells.keys():
                v['prior_probability'] = (
                        1 - all_probs) if all_probs > 0 else self.calc_estimated_prior_probability_for_cell(
                    cell_range=k,
                    all_probs=all_probs,
                    cell=v['cell'],
                    cell_info=v)  # If the sum of all probs is 0 (very low chance of it happening),
                # we calculate the probability using helper function.
            else:
                # Otherwise,
                # we're using the last value of the prior_probability in the next round before updating
                v['prior_probability'] = self._cells[k]['prior_probability']

        return cells_dict

    def calc_estimated_prior_probability_for_cell(self, cell_range, all_probs, cell, cell_info):
        # TODO: THINK HOW TO IMPLEMENT THIS FUNCTION!
        # IDEA:
        # GET SOME RANDOM (BUT CLOSE TO AVERAGE) POINT IN CELL_RANGE,
        # AND CALC ITS RATE, AND THE LUCE RATE, AND THIS WILL BE THE PRIOR PROBABILITY.
        pass

    def generate_random_reservation_points(self) -> list[tuple[float, tuple[float, float]]]:
        """
        Generate a single random reservation point for each cell according to on_agent condition.

        Return:
                reservation points in a list, where each element is a tuple of the cell index (first element)
                and the random reservation point itself (second element) represented by tuple too.
        """
        reservation_points = []
        for cell_key, cell_info in self._cells.items():
            t_low, t_high, p_low, p_high = cell_info['cell']  # Extract the boundaries from the tuple
            rand_rv = get_random_rv(time_lower_bound=t_low, time_upper_bound=t_high,
                                    price_lower_bound=p_low, price_upper_bound=p_high)
            reservation_points.append((cell_key, rand_rv))

        return reservation_points

    def update_probabilities(self, non_linear_correlation_coefficient, _round):
        for cell_key, correlation_coefficient_value in non_linear_correlation_coefficient.items():
            if cell_key in self._cells.keys():
                self.probabilities_for_each_cell[cell_key]['likelihood'] = correlation_coefficient_value

                print(
                    f"Probability for {cell_key}, correlation coefficient: {correlation_coefficient_value},"
                    f" self.probabilities_for_each_cell[cell_key]: {self.probabilities_for_each_cell[cell_key]}")
                # We update the posterior probabilities according to Bayes updating rule.
                self.probabilities_for_each_cell[cell_key]['posterior_probability'] = (correlation_coefficient_value *
                                                                                       self.probabilities_for_each_cell[
                                                                                           cell_key][
                                                                                           'prior_probability'])
                # Update the prior probability according to the posterior probability for the next rounds
                self.probabilities_for_each_cell[cell_key]['prior_probability'] = \
                    self.probabilities_for_each_cell[cell_key][
                        'posterior_probability']

        max_probabilities = max(
            self.probabilities_for_each_cell.items(), key=lambda item: custom_max(item)
        )  # For keeping the maximum probabilities.

        self.best_probabilities[_round] = max_probabilities
        print(f"At round {_round}, best_probability: {self.best_probabilities[_round]}")
        # TODO: SEE THE OUTPUT!!!
        raise Exception
