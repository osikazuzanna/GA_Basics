from audioop import cross
from mimetypes import init
from random import Random
from matplotlib.cbook import sanitize_sequence
from numpy.random import rand, randint 
import numpy as np
import matplotlib.pyplot as plt
from numpy import min, sum, ptp, array
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor #To retrain the surrogate model I used inceremental learning, since we want to keep some knowledge in each generation





def objective_function(I):
    min_func = 0.26*(I[0]**2 + I[1]**2) - 0.48*I[0]*I[1]
    max_func = 1/(1+min_func)

    return max_func


#Parameters

bounds = [[-10, 10], [-10, 10]]
iteration = 200
bits_per_var = 20
n_var = 2
pop_size = 100
crossover_rate = 0.7
mutation_rate = 0.3

#---------------------------------------------------------------------------


def init_pop(pop_size, bits_per_var, n_var):
    pop = np.zeros([pop_size, bits_per_var*2], dtype=int)
    for ind in range(pop_size):
        for var in range(n_var):
            for bit in range(bits_per_var):
                if (rand() <= 0.5):
                    pop[ind][(var+1)*bit] = 1
                else:
                    pop[ind][(var+1)*bit] = 0
    return pop


def crossover(pop, crossover_rate):
    offspring = np.zeros([pop_size, bits_per_var*n_var], dtype=int)
    off = 0

    while off < pop_size:
        parent1 = pop[randint(0, pop_size)]
        parent2 = pop[randint(0, pop_size)]

        if rand() < crossover_rate:
            offspring[off][:bits_per_var] = parent1[:bits_per_var]
            offspring[off][bits_per_var:] = parent2[bits_per_var]
            offspring[off+1][bits_per_var:] = parent1[bits_per_var]            
            offspring[off+1][:bits_per_var] = parent2[bits_per_var]

        else:
            offspring[off] = parent1
            offspring[off + 1] = parent2
        
        off += 2
    
    return offspring


def crossover_alt(pop, crossover_rate):
    offspring = np.empty((0, 40), dtype = int)

    for i in range(int(len(pop)/2)):
        p1 = pop[2*i-1].copy()
        p2 = pop[2*i].copy()

        if rand() < crossover_rate:
            cutting_point = randint(1, len(p1)-1, size = 2)
            while cutting_point[0] == cutting_point[1]:
                cutting_point = randint(1, len(p1) - 1, size = 2)
            cutting_point = sorted(cutting_point)
            c1 = np.array(list(p1[:cutting_point[0]]) + list(p2[cutting_point[0]:cutting_point[1]]) + list(p1[cutting_point[1]:]))
            c2 = np.array(list(p2[:cutting_point[0]]) + list(p1[cutting_point[0]:cutting_point[1]]) + list(p2[cutting_point[1]:]))
            offspring = np.vstack([offspring, c1, c2])
        else:
            offspring = np.vstack([offspring, p1, p2])
    
    return offspring




def mutation(offspring, mutation_rate):
    pop_offspring = offspring.copy()
    for ind in range(pop_size):
        for k in range(n_var*bits_per_var):
            if rand() < mutation_rate:
                if pop_offspring[ind][k] == 0:
                    pop_offspring[ind][k] = 1
                elif pop_offspring[ind][k] == 1:
                    pop_offspring[ind][k] = 0
    return pop_offspring


def mutation_alt(pop, mutation_rate):
    offspring = np.empty((0, 40), dtype = int)
    for i in range(int(len(pop))):
        p1 = pop[i].copy()
        if rand() < mutation_rate:
            cp = randint(0, len(p1))
            c1 = p1
            if c1[cp] == 1:
                c1[cp] = 0
            else:
                c1[cp] = 1
            offspring = np.vstack([offspring, c1])
        else:
            offspring = np.vstack([offspring, p1])
    return offspring





def decoding(bounds, bits_per_var, chromosome):
    real_chromosome = []
    for i in range(n_var):
        st, end = i*bits_per_var, (i*bits_per_var)+bits_per_var
        sub = chromosome[st:end]
        chars = ''.join([str(s) for s in sub])
        integer = int(chars, 2)
        real_value = bounds[i][0] + (integer/(2**bits_per_var))*(bounds[i][1] - bounds[i][0])
        real_chromosome.append(real_value)
    return real_chromosome


def selection(pop, fitness, pop_size):
    next_generation = np.zeros([pop_size, bits_per_var*n_var], dtype = int)
    elite = np.argmax(fitness)
    next_generation[0] = pop[elite]
    scaled_fitness = [(f-min(fitness))/ptp(fitness) for f in fitness]
    selection_prob = [f/sum(scaled_fitness) for f in scaled_fitness]
    index = list(range(pop_size*n_var))
    index_selected = np.random.choice(index, size = pop_size-1, replace = False, p = selection_prob)
    for i in range(1,pop_size):
        next_generation[i] = pop[index_selected[i-1]]
    
    return next_generation






#----------------------------------------------------------
#Initial population

pop = init_pop(pop_size, bits_per_var, n_var)
best_fitness = []
best_solution_encoded = []
best_solution_genotype = []



for gen in range(iteration):
    offspring = crossover_alt(pop, crossover_rate)
    offspring = mutation_alt(pop, mutation_rate)
    joint_pop = np.concatenate((pop, offspring))

    real_chromosome = [decoding(bounds, bits_per_var, p) for p in joint_pop]

    if gen % 5 == 0:
        fitness = [objective_function(real_values) for real_values in real_chromosome]
        surrogate_function = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(real_chromosome, fitness)
        index = np.argmax(fitness)
        current_best = joint_pop[index]
        best_solution_encoded.append(real_chromosome[index])
        best_solution_genotype.append(joint_pop[index])
        best_fitness.append(1/max(fitness) - 1)
        pop = selection(joint_pop, fitness, pop_size)
    else:
        fitness = surrogate_function.predict(real_chromosome)
        index = np.argmax(fitness)
        best_solution_encoded.append(real_chromosome[index])
        best_solution_genotype.append(joint_pop[index])
        best_fitness.append(1/max(fitness) - 1)
        pop = selection(joint_pop, fitness, pop_size)

print(best_fitness[199])
print(best_solution_encoded[199])
print(best_solution_genotype[199])
print(1/surrogate_function.predict([best_solution_encoded[199]]) - 1)
fig = plt.figure()
plt.plot(best_fitness)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
print('Min objective funtion value')
print('Optimal solution', decoding(bounds, bits_per_var, current_best))
plt.show()
