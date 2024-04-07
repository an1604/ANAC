from typing import Any
from cell import Cell
import numpy as np
from scipy.linalg._matfuncs import eps
from helpers_functions import get_random_rv, sigmoid
from sklearn.cluster import DBSCAN, KMeans

RANDOM_STATE = 42


def _range(start, stop, step):
    while start < stop:
        yield round(start, 10)
        start += step


class DetectionRegion:
    def __init__(
        self,
        n_clusters,
        deadline_time,
        initial_value,
        time_low_bound,
        first_play=False,
        reserved_value=None,
    ):
        self._first_play = first_play
        self._eps = eps
        self.deadline_time = deadline_time
        self.n_cells = n_clusters
        self._DetReg = {
            "time_low": time_low_bound,
            "time_high": 1.0,
            "initial_value": initial_value,
            "reserved_value": reserved_value if reserved_value is not None else 0.0,
        }
        self.Kmeans = KMeans(
            n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE
        )

        self._cells: list[Cell] = []
        self.initialize_cells_division(first_init=True)

        self.best_probabilities = []  # Keep track the best probability in each round

    def initialize_cells_division(self, first_init, X=None, X_labels=None):
        cells = []
        # In the first initialization, both X and X_label are None!
        if first_init or (X is None and X_labels is None):
            X = self.generate_X_offers()
            X_labels = self.Kmeans.fit_predict(X)

        for idx, cell_center in enumerate(self.Kmeans.cluster_centers_):
            # Calculate the boundaries based on the cell's cluster center
            time_low = cell_center[0] - (cell_center[0] - self._DetReg["time_low"]) / 2
            time_high = (
                cell_center[0] + (self._DetReg["time_high"] - cell_center[0]) / 2
            )
            value_low = (
                cell_center[1] - (cell_center[1] - self._DetReg["reserved_value"]) / 2
            )
            value_high = (
                cell_center[1] + (self._DetReg["initial_value"] - cell_center[1]) / 2
            )

            cell_range = np.array([time_low, time_high, value_low, value_high])

            # Checking what the probabilities are in each cell
            if first_init:
                # If we are in the first initialization, we randomly initialize the probabilities.
                prior_probability = 1 / self.n_cells
                likelihood = np.random.uniform(low=self._eps, high=2 * self._eps)
                posterior_probability = np.random.uniform(
                    low=self._eps, high=2 * self._eps
                )

            else:  # Means that self_cells already initialized
                contains, cell_associated = self.get_right_or_closest_cell(
                    cell_center, cell_range
                )
                if contains:
                    prior_probability = cell_associated.prob.prior_probability
                    likelihood = cell_associated.prob.likelihood_probability
                    posterior_probability = cell_associated.prob.posterior_probability
                else:  # If the cell was not in self._cells,
                    # we reduce a little the probability that the rv will be in that cell.
                    prior_probability = (
                        cell_associated.prob.prior_probability - self._eps
                    )
                    likelihood = cell_associated.prob.likelihood_probability - self._eps
                    posterior_probability = (
                        cell_associated.prob.posterior_probability - self._eps
                    )

            cell_key = self.get_cell_key(cell_center, idx)
            cells.append(
                Cell(
                    # Cluster information
                    cell_key=cell_key,
                    cluster_center=cell_center,
                    cell_index=idx,
                    cell_range=cell_range,
                    prior_probability=prior_probability,
                    likelihood=likelihood,
                    posterior_probability=posterior_probability,
                    # The offers that associated with the cluster.
                    offers_associated=np.array(
                        [X[i] for i in range(len(X)) if X_labels[i] == idx]
                    ),
                )
            )
        # Update the cells in self._cells, and return.
        self._cells = cells

    @staticmethod
    def get_cell_key(cell, idx):
        return f"{idx}_{cell}"

    def generate_X_offers(self, n_points=100000) -> np.array:
        time_low = self._DetReg["time_low"]
        time_high = self._DetReg["time_high"]
        reserved_value = self._DetReg["reserved_value"]
        initial_value = self._DetReg["initial_value"]

        # Uniformly initialize the offer's range.
        time_values = np.random.uniform(low=time_low, high=time_high, size=n_points)
        cluster_indices = np.arange(self.n_cells)
        np.random.shuffle(cluster_indices)
        cluster_ranges = np.linspace(time_low, time_high, self.n_cells + 1)

        offers = np.zeros((n_points, 2))
        points_per_cluster = n_points // self.n_cells
        point_index = 0
        for cluster_index in cluster_indices:
            # To make sure that all the cells will get equal offers in their range,
            # we iterate for each cell range that generated, and create offers equally.
            for _ in range(points_per_cluster):
                cluster_mean = (
                    cluster_ranges[cluster_index] + cluster_ranges[cluster_index + 1]
                ) / 2
                cluster_stdev = (
                    cluster_ranges[cluster_index + 1] - cluster_ranges[cluster_index]
                ) / 6
                time_val = time_values[point_index]
                value_val = np.random.normal(loc=cluster_mean, scale=cluster_stdev)
                value_val = max(min(value_val, initial_value), reserved_value)
                offers[point_index] = [time_val, value_val]
                point_index += 1

        return offers

    def set_first_play(self, first_play):
        self._first_play = first_play

    def set_lower_bound_time(self, lower_bound_time):
        self._DetReg["time_low"] = float(
            lower_bound_time / self.deadline_time
        )  # Normalization
        self.update()

    def set_reserved_value(self, reserved_value):
        self._DetReg["reserved_value"] = reserved_value
        self.update()

    def set_initial_value(self, initial_value):
        self._DetReg["initial_value"] = initial_value
        self.update()

    def update(self):
        # Getting the new bounds for the detection region
        t_low, t_high = self._DetReg["time_low"], self._DetReg["time_high"]
        p_low, p_high = self._DetReg["reserved_value"], self._DetReg["initial_value"]
        new_bounds = np.array([t_low, t_high, p_low, p_high])
        cells_in_boundaries = []
        for cell in self._cells:
            # If the current cell inside the new boundaries, we keep it.
            if cell.in_boundaries(new_boundaries=new_bounds):
                cells_in_boundaries.append(cell)

        # If we miss some clusters, we need to initialize new clusters as missed.
        if len(cells_in_boundaries) < self.n_cells:
            # Predicting the random generated points labels according to the new information.
            X = self.generate_X_offers()
            X_labels = self.Kmeans.predict(X)
            self.initialize_cells_division(
                first_init=False, X=X, X_labels=X_labels
            )  # Calling the cells initialization function
            # to update the cells according to the changes

        return  # Otherwise, we can return without update anything.

    def get_the_most_recent_cell_index(self, new_center_time):
        labels = self.Kmeans.labels_
        cluster_centers = self.Kmeans.cluster_centers_
        is_new_center_time_changed = False

        # Filter out cluster centers outside the time constraint
        valid_centers = [
            center
            for label, center in zip(labels, cluster_centers)
            if center[0] <= new_center_time
        ]
        if len(valid_centers) == 0:
            is_new_center_time_changed = True
            # If we found no such valid centers, we take the one with the minimum value of time.
            min_cluster = min(
                [center for center in cluster_centers], key=lambda x: x[0]
            )
            valid_centers = [min_cluster]

        # Calculate the distances between the valid cluster centers and the new time
        distances = [abs(center[0] - new_center_time) for center in valid_centers]

        # Find the index of the cluster center with the smallest distance
        min_distance_index = np.argmin(distances)

        # Get the label and corresponding cluster center
        most_recent_label = labels[min_distance_index]
        most_recent_center = valid_centers[min_distance_index]

        if is_new_center_time_changed:
            return most_recent_label, most_recent_center, most_recent_center[0]
        return most_recent_label, most_recent_center, None

    def get_cell_from_index(self, idx):
        nearest_cell = [cell for cell in self._cells if cell.cell_index == idx]
        if len(nearest_cell) == 0:
            return None
        return nearest_cell[0]

    def get_detection_region(self):
        return self._DetReg

    def get_best_cell(self, _round=None):
        return self.best_probabilities[-1][1]

    def get_estimated_rv(self) -> np.ndarray:
        best_cell = self.get_best_cell()
        t_low, t_high, p_low, p_high = best_cell.cell_range
        return np.array([(t_low + t_high) / 2.0, (p_low + p_high) / 2.0])

    def get_DetReg_bounds_in_ndarray(self) -> np.array:
        return np.array(
            [
                self._DetReg["time_low"],
                self._DetReg["time_high"],
                self._DetReg["initial_value"],
                self._DetReg["reserved_value"],
            ]
        )

    def generate_random_reservation_points(self) -> list[tuple[int, np.ndarray]]:
        """
        Generate a single random reservation point for each cell according to on_agent condition.

        Return:
                reservation points in a np.ndarray,
                where each element is a tuple of the cell index (first element)
                and the random reservation point itself (second element) represented by np.ndarray.
        """
        rand_choice = np.random.randint(low=0, high=2)
        if rand_choice in [0, 1]:
            return [
                tuple([cell.cell_index, cell.cluster_center] for cell in self._cells)
            ]
        else:
            return [
                tuple(
                    [
                        cell.cell_index,
                        get_random_rv(
                            time_lower_bound=cell.cell_range[0],
                            time_upper_bound=cell.cell_range[1],
                            price_lower_bound=cell.cell_range[2],
                            price_upper_bound=cell.cell_range[3],
                        ),
                    ]
                )
                for cell in self._cells
            ]

    def get_right_or_closest_cell(
        self, cell_center, cell_range=None
    ) -> tuple[bool, Cell]:
        """
        Checks if some cell center is in self._cell and returns it,
        and if not, returns the closest cell that kmeans predicts.
        """
        if (
            cell_center[0] > self._DetReg["time_low"]
            or cell_center[0] < self._DetReg["time_high"]
        ):
            for cell in self._cells:
                if (
                    cell_center[0] == cell.cluster_center[0]
                    and cell_center[1] == cell.cluster_center[1]
                ):
                    if cell_range is not None:
                        # We check if the ranges are equal too.
                        if (cell_range - cell.cell_range == 0).all():
                            return True, cell
                    return True, cell

        # If we are not found any cell, we cluster some offer in cell range and return the cell associated.
        offer_in_cell = get_random_rv(
            time_lower_bound=cell_range[0],
            time_upper_bound=cell_range[1],
            price_lower_bound=cell_center[2],
            price_upper_bound=cell_center[3],
        )
        cell_pred = self.Kmeans.predict([offer_in_cell])[
            0
        ]  # The nearest cell predicted.
        for cell in self._cells:
            if cell_pred == cell.cell_index:
                return False, cell

    def update_probabilities(
        self, non_linear_correlation_coefficient, _round, current_time
    ):
        for (
            cell_index,
            correlation_coefficient_value,
        ) in non_linear_correlation_coefficient.items():
            cell = self.get_cell_from_index(cell_index)
            if cell:
                cell.prob.update_likelihood_probability(
                    correlation_coefficient_value
                )  # Update the likelihood according to the correlation coefficient

                # We update the posterior probabilities according to Bayes updating rule.
                cell.prob.update_posterior_probability(
                    (correlation_coefficient_value * cell.prob.prior_probability)
                )
                self.switch(
                    cell
                )  # Switch the previous cell information with the updated information.

        # For keeping the maximum probabilities.
        max_probabilities = self.find_maximum_probability()
        self.best_probabilities.append(max_probabilities)

        for cell in self._cells:
            # Update the prior probability according to the posterior probability for the next rounds
            cell.prob.update_prior_probability(cell.prob.get_posterior_probability())

        # print(f"Probabilities after update at time {current_time}:")
        # for cell in self._cells:
        #     print(
        #         f"Cell information: {cell.__dict__},\n"
        #         f"\nfor cell {cell.cell_index}, with range: {cell.cell_range},"
        #         f"\nThe probabilities are: {cell.prob.__dict__} "
        #         f"\n\n"
        #     )

    def switch(self, cell):
        for _cell in self._cells:
            if _cell == cell:
                _cell = cell
                return

    def get_initial_value(self):
        return self._DetReg["initial_value"]

    def find_maximum_probability(self):
        max_probabilities = 0.0
        max_probability_cell = None
        for cell in self._cells:
            if cell.prob.posterior_probability > max_probabilities:
                max_probabilities = cell.prob.posterior_probability
                max_probability_cell = cell
        return max_probabilities, max_probability_cell
