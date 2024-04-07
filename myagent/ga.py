import math
import random
from numpy import sort

from myagent.helpers_functions import nash_optimally


class SINGLE_POP:
    def __init__(self, fitness_val, points_path):
        self.fitness_val = fitness_val
        self.points_path = points_path

    def __eq__(self, other):
        if isinstance(other, SINGLE_POP):
            return self.points_path == other.points_path and self.fitness_val == other.fitness_val
        return False


def dist(p1, p2, RESERVATION_VALUES):
    # Using the Nash optimally as evaluation distance function.
    return nash_optimally(utility1=p1.outcome, utility2=p2.outcome, rv1=RESERVATION_VALUES.self_rv,
                          rv2=RESERVATION_VALUES.opp_rv)


def fitness(points, RV):
    sum = 0
    for i in range(1, len(points)):
        sum += dist(points[i - 1], points[i], RV)
    return sum


def rand_chrom(points: list, len_points, RESERVATION_VALUES):
    p = points.copy()
    random.shuffle(p)
    while len(p) > len_points:
        p.remove(random.choice(p))
    p_fitness = fitness(p, RESERVATION_VALUES)
    return SINGLE_POP(fitness_val=p_fitness, points_path=p)


def crossover(parent1, parent2, len_points, RV):
    p1, p2 = parent1.points_path.copy(), parent2.points_path.copy()
    mid_p1 = len(p1) // 2
    mid_p2 = len(p2) // 2
    first_p = []
    for i in range(mid_p1):
        first_p.append(p1[i])
    for i in range(mid_p2, len_points):
        first_p.append(p2[i])
    first_p = rand_chrom(first_p, len_points, RV)

    second_p = []
    for i in range(mid_p2):
        second_p.append(p2[i])
    for i in range(mid_p1 + 1, len_points):
        second_p.append(p1[i])
    second_p = rand_chrom(second_p, len_points, RV)
    return first_p, second_p


def mutation(points, RV):
    new_p = points.copy()
    i, j = random.sample(range(len(points)), 2)
    new_p[i], new_p[j] = new_p[j], new_p[i]
    new_fitness_val = fitness(new_p, RV)
    mutated_individual = SINGLE_POP(fitness_val=new_fitness_val, points_path=new_p)
    return mutated_individual


def initialize_population(points, len_points, RV):
    population = []
    for i in range(len_points):
        p = rand_chrom(points, len_points, RV)
        if p is not None and len_points == len_points:
            population.append(p)
    population.sort(key=lambda x: x.fitness_val)
    return population, population[0]  # The arr and the best fitness value for now


def get_prob():
    return random.random()


def check_population(child, points, RV):
    # Check for missing points in the child chromosome
    if len(points) != len(child.points_path):
        present_points = set(child.points_path)
        missing_points = [p for p in points if p not in present_points]
        child_points_path = child.points_path + missing_points
        child.fitness_val = fitness(child_points_path, RV)
        child.points_path = child_points_path
    return child


def GA(points, muProb, RESERVATION_VALUES, generations=100):
    # PARAMS
    gen = 1
    len_points = len(points)

    # Step 1 - initialize random populations for start
    population, best_individual = initialize_population(points, len_points, RESERVATION_VALUES)
    while gen < generations:
        new_population = []
        for i in range(0, len(population), 2):
            # Get 2 parents for the next steps
            parent1, parent2 = population[i - 1], population[i]
            # Step 2 - cross over
            child1, child2 = crossover(parent1=parent2, parent2=parent2, len_points=len_points,
                                       RV=RESERVATION_VALUES)
            child1, child2 = check_population(child1, points, RESERVATION_VALUES), check_population(child2, points,
                                                                                                    RESERVATION_VALUES)

            # Step 3 - mutation
            if get_prob() < muProb:
                child1 = mutation(child1.points_path, RESERVATION_VALUES)
                child1 = check_population(child1, points, RESERVATION_VALUES)
            if get_prob() < muProb:
                child2 = mutation(child2.points_path, RESERVATION_VALUES)
                child2 = check_population(child2, points, RESERVATION_VALUES)
                # Adding the new childrens to the new list
            new_population.append(child1)
            new_population.append(child2)

        # population.extend(new_population)

        population = new_population
        population.sort(key=lambda p: p.fitness_val)

        # Check for the best fitness value each generation
        if best_individual.fitness_val > population[0].fitness_val:
            best_individual = population[0]

        gen += 1
    if best_individual.fitness_val > population[0].fitness_val:
        best_individual = population[0]
    return sorted(best_individual.points_path)[-1]  # To make sure we have no duplicate points in the best path


def solve(points, self_rv, opp_rv):
    return GA(points=points, muProb=0.5, RESERVATION_VALUES=RVS(self_rv, opp_rv))


class RVS:
    def __init__(self, self_rv, opp_rv):
        self.self_rv = self_rv
        self.opp_rv = opp_rv
