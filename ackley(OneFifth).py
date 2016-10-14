import math
import random
import sys

POPULATION_SIZE = 50
# Number of children
LAMBDA = 200
# Whether or not to use fixed 2 parents for each individual
FIXED_PARENTS = True
# Whether or not to use mu + lambda survivor selection
MU_PLUS_LAMBDA = False

# Constant step factor
C = 0.9

NUM_DIMENSIONS = 5
X_MIN = -15
X_MAX = 15

NUM_ITERATIONS = 1000

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
        child.append(float(xi + xj) / 2)
        if not FIXED_PARENTS:
            parents = select_parents(population)
    return child

def generate_offspring(population):
    offspring = list()
    for i in range(LAMBDA):
        child = recombination(population)
        offspring.append(child)
    return offspring

def mutation(child, Z):
    mutated = list()
    for i in range(NUM_DIMENSIONS):
        mutated.append(child[i] + Z[i])
    return child if ackley(child) <= ackley(mutated) else mutated

def perturbation(offspring, sigma):
    Z = [random.gauss(0, sigma) for i in range(NUM_DIMENSIONS)]
    new_offspring = list()
    s = 0
    t = 0
    for child in offspring:
        new_child = mutation(child, Z)
        new_offspring.append(new_child)
        if new_child != child:
            s += 1
        t += 1
    return (new_offspring, float(s) / t)
    
def adjust_sigma(sigma, ps):
    if ps > 0.2:
        return float(sigma) / C
    elif ps < 0.2:
        return sigma * C
    return sigma

def select_survivors(population, offspring):
    new_population = population + offspring if MU_PLUS_LAMBDA else offspring  
    new_population.sort(key = lambda x : ackley(x))
    new_population = new_population[:POPULATION_SIZE]
    return new_population

def check_for_solution(population):
    for x in population:
        # By using sys.float_info.epsilon, which equals to ~10e-16 in my machine,
        # the convergence rate is greatly diminished. Using 10e-6 instead.
        if abs(ackley(x)) < 10e-6:
            return True
    return False

def evolve():
    population = generate_population()
    sigma = 1.0
    for i in range(NUM_ITERATIONS):
        offspring = generate_offspring(population)
        (offspring, ps) = perturbation(offspring, sigma)
        sigma = adjust_sigma(sigma, ps)
        if check_for_solution(offspring):
            print "Solution found after " + str(i) + " iterations"
            return
        population = select_survivors(population, offspring)
        # DEBUG
        #print min(map(lambda x : ackley(x), population))
    print "No solution found after " + str(NUM_ITERATIONS) + " iterations"

evolve()