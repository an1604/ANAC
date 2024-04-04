import math
import random
from numpy import sort


class SINGLE_POP:
    def __init__(self, fitness_val, points_path):
        self.fitness_val = fitness_val
        self.points_path = points_path

    def __eq__(self, other):
        if isinstance(other, SINGLE_POP):
            return self.points_path == other.points_path and self.fitness_val == other.fitness_val
        return False


def dist(p1, p2):
    return abs(p1 - p2) if isinstance(p1, float) and isinstance(p2, float) \
        else abs(float(p1[-1]) - float(p2[-1]))


def fitness(points):
    sum = 0
    for i in range(1, len(points)):
        sum += dist(points[i - 1], points[i])
    return sum


def rand_chrom(points: list, len_points):
    p = points.copy()
    random.shuffle(p)
    while len(p) > len_points:
        p.remove(random.choice(p))
    p_fitness = fitness(p)
    return SINGLE_POP(fitness_val=p_fitness, points_path=p)


def crossover(parent1, parent2, len_points):
    p1, p2 = parent1.points_path.copy(), parent2.points_path.copy()
    mid_p1 = len(p1) // 2
    mid_p2 = len(p2) // 2
    first_p = []
    for i in range(mid_p1):
        first_p.append(p1[i])
    for i in range(mid_p2, len_points):
        first_p.append(p2[i])
    first_p = rand_chrom(first_p, len_points)

    second_p = []
    for i in range(mid_p2):
        second_p.append(p2[i])
    for i in range(mid_p1 + 1, len_points):
        second_p.append(p1[i])
    second_p = rand_chrom(second_p, len_points)
    return first_p, second_p


def mutation(points):
    new_p = points.copy()
    i, j = random.sample(range(len(points)), 2)
    new_p[i], new_p[j] = new_p[j], new_p[i]
    new_fitness_val = fitness(new_p)
    mutated_individual = SINGLE_POP(fitness_val=new_fitness_val, points_path=new_p)
    return mutated_individual


def initialize_population(points, len_points):
    population = []
    for i in range(len_points):
        p = rand_chrom(points, len_points)
        if p is not None and len_points == len_points:
            population.append(p)
    population.sort(key=lambda x: x.fitness_val)
    return population, population[0]  # The arr and the best fitness value for now


def get_prob():
    return random.random()


def check_population(child, points):
    # Check for missing points in the child chromosome
    if len(points) != len(child.points_path):
        present_points = set(child.points_path)
        missing_points = [p for p in points if p not in present_points]
        child_points_path = child.points_path + missing_points
        child.fitness_val = fitness(child_points_path)
        child.points_path = child_points_path
    return child


def GA(points, muProb, generations=100):
    # PARAMS
    gen = 1
    len_points = len(points)

    # Step 1 - initialize random populations for start
    population, best_individual = initialize_population(points, len_points)
    while gen < generations:
        new_population = []
        for i in range(0, len(population), 2):
            # Get 2 parents for the next steps
            parent1, parent2 = population[i - 1], population[i]
            # Step 2 - cross over
            child1, child2 = crossover(parent1=parent2, parent2=parent2, len_points=len_points)
            child1, child2 = check_population(child1, points), check_population(child2, points)

            # Step 3 - mutation
            if get_prob() < muProb:
                child1 = mutation(child1.points_path)
                child1 = check_population(child1, points)
            if get_prob() < muProb:
                child2 = mutation(child2.points_path)
                child2 = check_population(child2, points)
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


def solve(points):
    return GA(points=points, muProb=0.5)
