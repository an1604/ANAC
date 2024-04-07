import numpy as np
import scipy.stats as stats

from myagent.cell import Cell


class Probabilities:
    def __init__(self, prior, likelihood, posterior, Y, cell):
        self.cell = cell  # The cell's associated with the probabilities.
        self.prior_probability = prior
        self.likelihood_probability = likelihood
        self.posterior_probability = posterior
        self.can_sd = 0.05  # The standard deviation for sampling from normal distribution.
        self.Y = Y

    def get_cell(self):
        return self.cell

    def update_prior_probability(self, new_value):
        self.prior_probability = new_value

    def get_prior_probability(self):
        return self.prior_probability

    def get_likelihood_probability(self):
        return self.likelihood_probability

    def update_posterior_probability(self, new_value):
        self.posterior_probability = new_value

    def update_likelihood_probability(self, new_value):
        self.likelihood_probability = new_value

    def get_posterior_probability(self):
        return self.posterior_probability

    def init_probs(self, Y):
        pass

    def test_for_standard_deviation(self, test_point):
        return stats.norm(test_point, self.can_sd).pdf(1)


def get_beta_distribution(cell, alpha=1, beta=1):
    samples = np.random.beta(alpha, beta, size=len(cell.offers_associated))
    samples = samples / np.sum(samples)
    return samples


class Learner:
    """
    The learner class is responsible for learning the opponent's behavior
    by assign prior probabilities to each cell to include the reservation value (rv) of the opponent.
    And at each round, the target is to precisely update the beliefs of the opponent's rv
    (the likelihood and the posterior probability),
    according to the prior probability distribution that assigned in the previous round.
    """

    def __init__(self, cells: list[Cell], n_cells):
        self._n_cells = n_cells
        self._cells_with_probs = self.init_probs(cells)
        self._cells = cells

    def init_probs(self, cells: list[Cell]):
        """
        Given cells' list,
        we initialize the probabilities of each cell according to the following conditions:
            1) Initialize prior probabilities for each cell as the HG probability distribution.
            2)
        """
        probs = []
        for cell in cells:
            # Initialize beta distribution over all offers
            # associated with the current cell and normalized the probabilities.
            prior = self.distribution_mass_function(cell)
            p_cell = {
                cell.cell_key: Probabilities(
                    # REALLY-REALLY FIRST IMPLEMENTATION!
                    cell=cell,
                    prior=prior,  # TODO: CHECK THIS OUT
                    likelihood=None,
                    posterior=None,
                    Y=cell.offers_associated
                )}
            probs.append(p_cell)
        if len(probs) == self._n_cells:
            return probs

    def update_probabilities(self, non_linear_correlation_coefficient, _round, current_time):
        cells_updated = []
        for cell_index, correlation_coefficient_value in non_linear_correlation_coefficient.items():
            cell = self.get_cell_from_index(cell_index)
            if cell:
                cell_key = cell.cell_key
                cell_proba = self._cells_with_probs[cell_key].get(cell_key)
                if cell_proba:  # Check if the cell found in the list.
                    # Calculate the new posterior probability.
                    new_posterior = correlation_coefficient_value * cell_proba.prior_probability

                    # Update our beliefs according to the correlation coefficient and the new_posterior values.
                    cell_proba.update_likelihood_probability(correlation_coefficient_value)
                    cell_proba.update_posterior_probability(new_posterior)
                    cell_proba.update_prior_probability(new_posterior)

                    cells_updated.append(cell_proba)

        # Set the class list of cell's probabilities to the updated one.
        if len(cells_updated) == self._n_cells:
            self._cells_with_probs = cells_updated

    def get_cell_from_index(self, cell_index):
        nearest_cell = [cell for cell in self._cells if cell.cell_index == cell_index]
        if len(nearest_cell) == 0:
            return None
        return nearest_cell[0]

    def get_max_probs_cell(self):
        """Getting the cell with the highest posterior probability in the list."""
        cells_sorted = sorted(self._cells_with_probs, key=lambda cell_k, cell_probs: cell_probs.posterior_probability,
                              reverse=True)  # Sorting the elements by their posterior probability values.
        max_prob_cell_key, max_prob_cell_value = cells_sorted[0]
        return max_prob_cell_value

    @staticmethod
    def distribution_mass_function(cell):
        """Share probability distribution according to the given cell under some conditions."""
        t_low, t_high = cell.cell_range[0], cell.cell_range[1]
        p_low, p_high = cell.cell_range[2], cell.cell_range[3]

        if t_low < 0 or p_high > 1:  # Out-of-bounds scenario
            return 0.

        elif (p_low + p_high) / 2 > 0.5:  # Average higher price scenario
            if (t_low + t_high) / 2 >= 0.5:  # Average higher time scenario
                return 0.2
            else:  # Average lower time scenario
                return 0.0333

        else:  # Average lower price scenario
            if (t_low + t_high) / 2 >= 0.5:  # Average higher time scenario
                return 0.25
            else:  # Average lower time scenario
                return 0.066
