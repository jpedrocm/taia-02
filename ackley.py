import math
import random

POPULATION_SIZE = 100
NUM_DIMENSIONS = 30
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