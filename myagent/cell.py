import numpy as np
from scipy import stats


class Cell:
    def __init__(self, cluster_center: np.ndarray, cell_index, cell_range: np.ndarray, offers_associated: np.ndarray,
                 cell_key, prior_probability, likelihood, posterior_probability):
        self.cluster_center = cluster_center
        self.cell_index = cell_index
        self.cell_range = cell_range
        self.offers_associated = offers_associated
        self.cell_key = cell_key
        self.prob = Probabilities(prior_probability, likelihood,
                                  posterior_probability, Y=offers_associated)

    def offer_in_cell(self, offer):
        return True

    def in_boundaries(self, new_boundaries: np.ndarray):
        """Checks if the cell is in the detection region new boundaries."""
        return ((new_boundaries - self.cell_range) >= 0).all()

    def print(self):
        print(f'{self.__dict__}\nWith probs:{self.prob.__dict__}\n')

    def __eq__(self, other):
        return (self.cell_index == other.cell_index and
                self.cell_range[0] == other.cell_range[0] and
                self.cell_range[1] == other.cell_range[1] and
                self.cluster_center[0] == other.cluster_center[0] and
                self.cluster_center[1] == other.cluster_center[1] and
                self.cell_key == other.cell_key
                )


class Probabilities:
    def __init__(self, prior, likelihood, posterior, Y):
        self.prior_probability = prior
        self.likelihood_probability = likelihood
        self.posterior_probability = posterior
        self.can_sd = 0.05  # The standard deviation for sampling from normal distribution.
        self.Y = self.init_probs(Y)  # Some samples

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
