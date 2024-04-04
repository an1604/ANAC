import random

import numpy as np


def get_random_rv(time_lower_bound, time_upper_bound, price_lower_bound, price_upper_bound):
    rand_time = random.uniform(time_lower_bound, time_upper_bound)
    rand_price = random.uniform(price_lower_bound, price_upper_bound)
    return rand_time, rand_price


def custom_max(item):
    return (
        item[1]['posterior_probability'],  # Phase 1: Compare by posterior probability
        item[1]['likelihood'],  # Phase 2: Compare by likelihood
        item[1]['prior_probability']  # Phase 3: Compare by prior probability
    )


def custom_max_offers(item):
    return (
        item[1]['ranking_rate'],
        item[1]['outcome'],
        item[1]['lucia_rate']
    )


def print_situation(situation):
    for k, v in situation.items():
        print(f'{k}: {v}\n')


def nash_optimality(utility1, rv1, utility2, rv2):
    return (utility1 - rv1) * (utility2 - rv2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
