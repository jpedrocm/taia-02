import math
import random
import sys

POPULATION_SIZE = 50
# Number of children
LAMBDA = 100
# Whether or not to use fixed 2 parents for each individual
FIXED_PARENTS = True
# Whether or not to use mu + lambda survivor selection
MU_PLUS_LAMBDA = False
# Number of dimensions
NUM_DIMENSIONS = 30
# If it should or should not use recombination
USE_RECOMBINATION = True
# Constant step factor
TAL_LOCAL = 1/math.sqrt(2*math.sqrt(NUM_DIMENSIONS))
TAL_GLOBAL = 1/math.sqrt(2*NUM_DIMENSIONS)


X_MIN = -15
X_MAX = 15

NUM_ITERATIONS = 1000

def generate_population():
    population = list()
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            #The first element of the tuple is the genome value
            #The second element of the tuple is the std associated with the pace for that genome
            tup1 = (random.uniform(X_MIN, X_MAX), 1);
            genome.append(tup1)
        population.append(genome)
    return population

def ackley(genome):
    n = len(genome)

    sum1 = sum(map(lambda x : x[0] ** 2, genome))
    sum2 = sum(map(lambda x : math.cos(2 * math.pi * x[0]), genome))
    a = -0.2 * math.sqrt((1.0 / n) * sum1)
    b = (1.0 / n) * sum2
    return -20 * math.exp(a) - math.exp(b) + 20 + math.e

def select_parents(population):
    return random.sample(population, 2)

def recombination(population):
    parents = select_parents(population)
    child = list()
    for i in range(NUM_DIMENSIONS):
        xi = parents[0][i][0]
        xj = parents[1][i][0]
        stdi = parents[0][i][1]
        stdj = parents[1][i][1]
        child.append(((float(xi + xj) / 2), (float(stdi + stdj) / 2)))
        if not FIXED_PARENTS:
            parents = select_parents(population)
    return child

def generate_offspring(population):
    offspring = list()
    for i in range(LAMBDA):
        child = recombination(population)
        offspring.append(child)
    return offspring

def mutation(child):
    mutated = list()
    Z = [random.gauss(0, child[i][1]) for i in range(NUM_DIMENSIONS)]
    for i in range(NUM_DIMENSIONS):
        mutated.append(((child[i][0] + Z[i]),child[i][1]))
    
    return child if ackley(child) <= ackley(mutated) else mutated

def perturbation(population):
    new_offspring = list()
    s = 0
    t = 0
    global_adjustment = TAL_GLOBAL * random.gauss(0,1)
    for child in population:
        new_child = adjust_sigma(child,global_adjustment)
        new_child = mutation(new_child)
        new_offspring.append(new_child)
        if new_child != child[0]:
            s += 1
        t += 1
    return (new_offspring, float(s) / t)
    
def adjust_sigma(individual,global_adjustment):
    newIndividual = list()
    for i in range(NUM_DIMENSIONS):
        sigma = individual[i][0]
        newSigma = sigma * math.exp(TAL_LOCAL * random.gauss(0,1) + global_adjustment)
        newIndividual.append((individual[i][0],newSigma))
    return newIndividual
    

def select_survivors(population, offspring):
    new_population = population + offspring if MU_PLUS_LAMBDA else offspring  
    new_population.sort(key = lambda x : ackley(x))
    new_population = new_population[:POPULATION_SIZE]
    return new_population

def check_for_solution(population):
    for x in population:
        # By using sys.float_info.epsilon, which equals to ~10e-16 in my machine,
        # the convergence rate is greatly diminished. Using 10e-6 instead.
        if abs(ackley(x)) < 10e-10:
            return True
    return False

def evolve():
    population = generate_population()
    #sigma = 1.0
    for i in range(NUM_ITERATIONS):
        offspring = list()
        if USE_RECOMBINATION:
            offspring = generate_offspring(population)
            
        (offspring, ps) = perturbation(population + offspring)
        if check_for_solution(offspring):
            print "Solution found after " + str(i) + " iterations"
            return
        population = select_survivors(population, offspring)
        # DEBUG
        #print min(map(lambda x : ackley(x), population))
    print "No solution found after " + str(NUM_ITERATIONS) + " iterations"

evolve()