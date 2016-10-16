import math
import random
import numpy
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
ZERO_CORRELATION_PROBABILITY = 0.5
# 5 degrees as slides specify
BETA = math.pi / 36

X_MIN = -15
X_MAX = 15

NUM_ITERATIONS = 1000

ALFA_MAPPING = []

def generate_population():
    population = list()
    alfa = [0] * (NUM_DIMENSIONS*(NUM_DIMENSIONS - 1)/2)
    for i in range(POPULATION_SIZE):
        genome = list()
        for s in range(NUM_DIMENSIONS):
            #The first element of the tuple is the genome value
            #The second element of the tuple is the std associated with the pace for that genome
            tup1 = (random.uniform(X_MIN, X_MAX), 1);
            genome.append(tup1)
        population.append((genome, alfa))
    return population

def covarianceMat(sigmas,alfas):
    covMatrix = list()
    for i in range(NUM_DIMENSIONS):
        line = list()
        for j in range(NUM_DIMENSIONS):
            if i == j:
                line.append(sigmas[j][1] * sigmas[j][1])
            else:
                value = (sigmas[j][1] * sigmas[j][1] - sigmas[i][1] * sigmas[i][1])/2
                value *= math.tan(2*alfas[ALFA_MAPPING[i][j]])
                line.append(value)
        covMatrix.append(line)
    return covMatrix
                

def perturbationForCov(cov,mean):
    length = len(cov)
    meanList = [mean]*length
    # print numpy.random.multivariate_normal(meanList,cov)
    return numpy.random.multivariate_normal(meanList,cov)

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
    genome = list()
    for i in range(NUM_DIMENSIONS):
        xi = parents[0][0][i][0]
        xj = parents[1][0][i][0]
        stdi = parents[0][0][i][1]
        stdj = parents[1][0][i][1]
        genome.append(((float(xi + xj) / 2), (float(stdi + stdj) / 2)))
        if not FIXED_PARENTS:
            parents = select_parents(population)
    
    alfas = list()
    alfa_sz = (NUM_DIMENSIONS*(NUM_DIMENSIONS - 1)/2)
    for i in range(alfa_sz):
        alfas.append((parents[1][1][i]+parents[0][1][i])/2)

    return (genome,alfas)

def generate_offspring(population):
    offspring = list()
    for i in range(LAMBDA):
        child = recombination(population)
        offspring.append(child)
    return offspring

def mutation(genome, alfas):
    mutated = list()
    matrix = covarianceMat(genome,alfas)
    Z = perturbationForCov(matrix,0)
    for i in range(NUM_DIMENSIONS):
        mutated.append(((genome[i][0] + Z[i]),genome[i][1]))
    
    # print "------"
    # print Z
    # print ackley(genome)
    # print ackley(mutated)
    return genome if ackley(genome) <= ackley(mutated) else mutated

def perturbation(population):
    new_offspring = list()
    s = 0
    t = 0
    global_adjustment = TAL_GLOBAL * random.gauss(0,1)
    for child in population:
        genome = child[0]
        alfas = child[1]
        genome_adjusted_sigmas = adjust_sigma(genome,global_adjustment)
        new_alfas = adjust_alfa(alfas)
        genome_adjusted_sigmas = mutation(genome_adjusted_sigmas, new_alfas)
        
        new_offspring.append((genome_adjusted_sigmas, new_alfas))
        
    return new_offspring
    
def adjust_alfa(alfas):
    newAlfa = list()
    for i in alfas:
        if random.uniform(0,1) > ZERO_CORRELATION_PROBABILITY:
            na = i + BETA * random.gauss(0,1)
            if (abs(na) > math.pi):
                na = na - 2 * math.pi * numpy.sign(na)
            newAlfa.append(na)

        else :
            newAlfa.append(0)

    # print"aaaaaaaaa\n"
    # print newAlfa
    return newAlfa
    
def adjust_sigma(individual,global_adjustment):
    newIndividual = list()
    for i in range(NUM_DIMENSIONS):
        sigma = individual[i][0]
        newSigma = sigma * math.exp(TAL_LOCAL * random.gauss(0,1) + global_adjustment)
        newIndividual.append((individual[i][0],newSigma))
    return newIndividual

def select_survivors(population, offspring):
    new_population = population + offspring if MU_PLUS_LAMBDA else offspring  
    new_population.sort(key = lambda x : ackley(x[0]))
    new_population = new_population[:POPULATION_SIZE]
    return new_population

def check_for_solution(population):
    for x in population:
        # By using sys.float_info.epsilon, which equals to ~10e-16 in my machine,
        # the convergence rate is greatly diminished.
        if abs(ackley(x[0])) < 10e-10:
            return True
    return False

def pre_calc_alpha_mapping():
    global ALFA_MAPPING
    ALFA_MAPPING = [([(-1) for y in range(NUM_DIMENSIONS)]) for x in range(NUM_DIMENSIONS)]
    counter = 0
    for i in range(NUM_DIMENSIONS):
        for j in range(NUM_DIMENSIONS):
            if i != j:
                if ALFA_MAPPING[j][i] == -1:
                    ALFA_MAPPING[i][j] = counter
                    counter += 1
                else:
                    ALFA_MAPPING[i][j] = ALFA_MAPPING[j][i]                  

def evolve():
    population = generate_population()
    pre_calc_alpha_mapping()
    for i in range(NUM_ITERATIONS):
        offspring = list()
        if USE_RECOMBINATION:
            offspring = generate_offspring(population)
            
        offspring = perturbation(population + offspring)
        if check_for_solution(offspring):
            print "Solution found after " + str(i) + " iterations"
            return
        #print "######"
        #print min(map(lambda x : ackley(x[0]), population))
        #print min(map(lambda x : ackley(x[0]), offspring))
        population = select_survivors(population, offspring)
        # DEBUG
        #print min(map(lambda x : ackley(x[0]), population))

    print "No solution found after " + str(NUM_ITERATIONS) + " iterations"
    # print population

evolve()
#pre_calc_alpha_mapping()