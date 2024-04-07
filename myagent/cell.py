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


