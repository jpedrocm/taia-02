import math
import random

POPULATION_SIZE = 10
LAMBDA = 20
FIXED_PARENTS = True

NUM_DIMENSIONS = 3
X_MIN = -15
X_MAX = 15

def generate_population():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            genome.append(random.uniform(X_MIN, X_MAX))
        population.append(genome)
    return population

def ackley(genome):
    n = len(genome)
    sum1 = sum(map(lambda x : x ** 2, genome))
    sum2 = sum(map(lambda x : math.cos(2 * math.pi * x), genome))
    a = -0.2 * math.sqrt((1.0 / n) * sum1)
    b = (1.0 / n) * sum2
    return -20 * math.exp(a) - math.exp(b) + 20 + math.e

def select_parents(population):
    return random.sample(population, 2)

def recombination(population):
    parents = select_parents(population)
    child = list()
    for i in range(NUM_DIMENSIONS):
        xi = parents[0][i]
        xj = parents[1][i]
        child.append((xi + xj) / 2)
        if not FIXED_PARENTS:
            parents = select_parents(population)
    return child

def generate_offspring(population):
    offspring = list()
    for i in range(LAMBDA):
        child = recombination(population)
        offspring.append(child)
    return offspring

population = generate_population()