import math

from negmas import OutcomeSpace
from negmas.preferences import UFun

"""
    This file is about checking similarities between the decisional values of the agents.
    In this file, we are comparing the two agents utility functions values,preferences, and outcome spaces
    to determine whether they have similar satisfactions, and later to cluster the opponents to adjust our's 
    agent behavior.   
"""


def compare_utilities(self_ufun: UFun, other_ufun: UFun, self_outcome_spce: OutcomeSpace) -> float:
    """
    Compares utilities based on the self agent's outcome space and minimizes the epsilon value.
    This function implements the condition:
    ui ≈ uj ⇒ ∀x ∈ D, ∃ε, |ui(x) - uj(x)| ≤ ε
    where ui and uj are utility functions, x is an outcome, and ε is the minimum value.

    Args:
        self_ufun: The self agent's utility function.
        other_ufun: The other agent's utility function.
        self_outcome_spce: The self agent's outcome space.

    Returns:
        The minimum epsilon value required for all outcomes in the self agent's outcome space
        to be similar (within epsilon) to the other agent's utility for those outcomes.
        Returns -1.0 if no such epsilon exists.

    ***
    Note: we need to agree on some generic epsilon (threshold) to make the decision if two utilities are similar or not.
    ***
    """
    if self_outcome_spce is None:
        return -1.0

    min_epsilon = float('inf')

    for out in self_outcome_spce:
        for epsilon in (x / 1000.0 for x in range(1, 1000)):  # Iterate through epsilon values with precision
            difference = abs(self_ufun(out) - other_ufun(out))
            if difference > epsilon:
                min_epsilon = min(min_epsilon, difference)
                break
    return min_epsilon if min_epsilon != float('inf') else -1.0


from scipy import integrate


def sim(self_ufun: UFun, other_ufun: UFun, self_outcome_spce: OutcomeSpace,
        min_x: float, max_x: float) -> float:
    """
      Compares utilities based on the self agent's outcome space and calculates the integral of the difference
      between utilities within the specified range.

      Args:
          self_ufun: The self agent's utility function.
          other_ufun: The other agent's utility function.
          self_outcome_spce: The self agent's outcome space.
          min_x: The minimum value for the integration range (inclusive).
          max_x: The maximum value for the integration range (inclusive).

      Returns:
          The integral of the absolute difference between self and other agent's utility within the outcome space
          and the specified range. Returns -1.0 if an error occurs during integration.

    ***
    Note: We need to think about a threshold too.
    ***
      """
    if self_outcome_spce is None:
        return -1.0
    # Check if the utilities is exactly the same
    min_epsilon = compare_utilities(self_ufun, other_ufun, self_outcome_spce)
    if min_epsilon == 0.0:
        return 0.0

    # Define the integration function
    def integrand(x):
        return abs(self_ufun(x) - other_ufun(x))

    # Attempt integration
    try:
        integral_value, _ = integrate.quad(integrand, min_x, max_x)
        return integral_value
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1.0


def beliefs(self_ufun: UFun, other_ufun: UFun,
            possible_values: OutcomeSpace, self_beliefs=None,
            other_beliefs=None):
    """
    Two agent i and j will share the same certainties (beliefs) for an attribute if their respective
    probability distributions over the attribute are close or similar.
    A possible way to consider this similarity is to use the cross-entropy.
    Args:
        self_ufun (UFun): The utility function for agent 1.
        other_ufun (Ufun): The utility function for agent 2
        possible_values: A list of all possible values for the attribute.
        self_beliefs (Optional): Initial probability distribution for agent 1 (default: uniform probability distribution).
        other_beliefs (Optional): Initial probability distribution for agent 2 (default:uniform probability distribution)
     Returns:
      A tuple containing the belief of agent 1 about agent 2's preference (float)
      and the belief of agent 2 about agent 1's preference (float).
    """
    # Initialize uniform probability distributions if not provided
    if self_beliefs is None:
        self_beliefs = [1.0 / len(possible_values) * len(possible_values)]
    if other_beliefs is None:
        other_beliefs = [1.0 / len(possible_values) * len(possible_values)]

    # Calculate cross-entropy for each agent
    agent1_entropy = 0.0
    agent2_entropy = 0.0
    for i, value in enumerate(possible_values):
        agent1_utility = self_ufun(value)
        agent2_prob = other_beliefs[i]
        agent1_entropy += -agent2_prob * math.log(agent1_utility)

        agent2_utility = other_ufun(value)
        agent1_prob = self_beliefs[i]
        agent2_entropy += -agent1_prob * math.log(agent2_utility)
    return -agent1_entropy, -agent2_entropy

# TODO: Implement the DBSCAN clustering technique to cluster a new agent and learn his approach according to
#  previous negotiations
# def dbscan_clustering(agents_dic:dict, epsilon:float, ):
