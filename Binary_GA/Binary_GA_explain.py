from audioop import cross
from mimetypes import init
from matplotlib.cbook import sanitize_sequence
from numpy.random import rand, randint 
import numpy as np
import matplotlib.pyplot as plt
from numpy import min, sum, ptp, array
from sklearn.linear_model import LinearRegression, SGDRegressor #To retrain the surrogate model I used inceremental learning, since we want to keep some knowledge in each generation
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import pandas as pd
import sklearn.ensemble
import statistics




#Parameters

bounds = [[-10, 10], [-10, 10]]
iteration = 200
bits_per_var = 20
n_var = 2
pop_size = 100
crossover_rate = 0.7
mutation_rate = 0.3

#---------------------------------------------------------------------------

class Binary_GA:

#Constructs a new genetic algorithm object
# @param bounds refers to variable bounds
# @param interation refers to number of generations
# @param bits_per_var refers to the number of bits per every variable
# @param n_var refers to number of variables
# @param pop_size refers to a population size in each generation
# @param crossover_rate is a probability of crossover
# @param mutation_rate is a mutation rate

    def __init__(self, bounds, iteration, bits_per_var, n_var, pop_size, crossover_rate, mutation_rate, surrogate = None):
        self.bounds = bounds
        self.iteration = iteration
        self.bits_per_var = bits_per_var
        self.n_var = n_var
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.surrogate = surrogate
    
    def objective_function(self, I):
        x = I[0]
        y = I[1]
        min_func = 0.26*(x**2 + y**2) - 0.48*x*y
        max_func = 1/(1+min_func)
        return max_func


      #initiate population
    def init_pop(self):
        pop = np.zeros([self.pop_size, self.bits_per_var*2], dtype=int)
        for ind in range(self.pop_size):
            for var in range(self.n_var):
                for bit in range(self.bits_per_var):
                    if (rand() <= 0.5):
                        pop[ind][(var+1)*bit] = 1
                    else:
                        pop[ind][(var+1)*bit] = 0
        return pop
     
     #crossover
    def crossover(self, pop):
        offspring = np.empty((0, 40), dtype = int)

        for i in range(int(len(pop)/2)):
            p1 = pop[2*i-1].copy()
            p2 = pop[2*i].copy()

            if rand() < self.crossover_rate:
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

     #mutation
    def mutation(self, pop):
        offspring = np.empty((0, 40), dtype = int)
        for i in range(int(len(pop))):
            p1 = pop[i].copy()
            if rand() < self.mutation_rate:
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
        
     #decode binary strings to real values
    def decoding(self, chromosome):
        real_chromosome = []
        for i in range(self.n_var):
            st, end = i*self.bits_per_var, (i*self.bits_per_var)+self.bits_per_var
            sub = chromosome[st:end]
            chars = ''.join([str(s) for s in sub])
            integer = int(chars, 2)
            real_value = bounds[i][0] + (integer/(2**self.bits_per_var))*(self.bounds[i][1] - self.bounds[i][0])
            real_chromosome.append(real_value)
        return real_chromosome

     #selection of pop_size best solutions based onwheel roulette
    def selection(self, pop, fitness):
        next_generation = np.zeros([self.pop_size, self.bits_per_var*n_var], dtype = int)
        elite = np.argmax(fitness)
        next_generation[0] = pop[elite]
        scaled_fitness = [(f-min(fitness))/ptp(fitness) for f in fitness]
        selection_prob = [f/sum(scaled_fitness) for f in scaled_fitness]
        index = list(range(self.pop_size*2))
        index_selected = np.random.choice(index, size = self.pop_size-1, replace = False, p = selection_prob)
        for i in range(1,self.pop_size):
            next_generation[i] = pop[index_selected[i-1]]
        return next_generation

    def solve(self):
        pop = self.init_pop()
        best_fitness = []
        best_solution_encoded = []
        best_solution_genotype = []
        for gen in range(self.iteration):
            offspring = self.crossover(pop)
            offspring = self.mutation(pop)
            joint_pop = np.concatenate((pop, offspring))
            real_chromosome = [self.decoding(p) for p in joint_pop]
            if self.surrogate == None:
                fitness = [self.objective_function(real_values) for real_values in real_chromosome]
                index = np.argmax(fitness)
                best_solution_encoded.append(real_chromosome[index])
                best_solution_genotype.append(joint_pop[index])
                best_fitness.append(1/max(fitness) - 1)
                pop = self.selection(joint_pop, fitness)
            else:
                if gen % 5 == 0:
                    fitness = [self.objective_function(real_values) for real_values in real_chromosome]
                    if self.surrogate == 'RandomForest':
                        surrogate_function = RandomForestRegressor().fit(real_chromosome, fitness)
                    if self.surrogate == 'LinearRegression':
                        surrogate_function = LinearRegression().fit(real_chromosome, fitness)
                    index = np.argmax(fitness)
                    best_solution_encoded.append(real_chromosome[index])
                    best_solution_genotype.append(joint_pop[index])
                    best_fitness.append(1/max(fitness) - 1)
                    pop = self.selection(joint_pop, fitness)
                else:
                    fitness = surrogate_function.predict(real_chromosome)
                    index = np.argmax(fitness)
                    best_solution_encoded.append(real_chromosome[index])
                    best_solution_genotype.append(joint_pop[index])
                    best_fitness.append(1/max(fitness) - 1)
                    pop = self.selection(joint_pop, fitness)

        if self.surrogate == None:
            return best_fitness, best_solution_encoded, best_solution_genotype
        else:
            return best_fitness, best_solution_encoded, best_solution_genotype, surrogate_function
        
    def importance(self, n_explain):

        cumulated_importance = []
        for _ in range(n_explain):
            importance = []
            fitness, phenotype, genotype, surrogate_function = self.solve()    
            for i in range(self.n_var):
                pert_solution = genotype[len(fitness)-1].copy()
                cut_point = randint(bits_per_var*i, bits_per_var*(i+1))
                if pert_solution[cut_point] == 0:
                    pert_solution[cut_point] = 1
                else:
                    pert_solution[cut_point] = 0

                real_chromosome = []

                for j in range(n_var):
                    st, end = j*bits_per_var, (j*bits_per_var) + bits_per_var
                    sub = pert_solution[st:end]
                    chars = ''.join([str(s) for s in sub])
                    integer = int(chars, 2)
                    real_value = bounds[j][0] + (integer/(2**bits_per_var))*(bounds[j][1] - bounds[j][0])
                    real_chromosome.append(real_value)

                new_fitness = 1/surrogate_function.predict([real_chromosome]) - 1
                change_i = float(abs(fitness[len(fitness) - 1] - new_fitness))
                importance.append(change_i)

            cumulated_importance.append(importance)

        final_importance = []
        for i in range(self.n_var):
            vi_importance = statistics.median([imp[i] for imp in cumulated_importance])
            final_importance.append(vi_importance)

        return pert_solution, cumulated_importance, final_importance










#----------------------------------------------------------
#Initial population

if __name__ == '__main__':
    #parameters
    bounds = [[-10, 10], [-10, 10]]
    iteration = 100
    bits_per_var = 20
    n_var = 2
    pop_size = 100
    crossover_rate = 0.7
    mutation_rate = 0.2
    # print('Min objective funtion value')
    # print('Optimal solution', decoding(bounds, bits_per_var, current_best))
    ga = Binary_GA(bounds, iteration, bits_per_var, n_var, pop_size, crossover_rate, mutation_rate, surrogate = 'RandomForest')
    fitness, phenotype, genotype, surrogate = ga.solve()
    pert_solution, cumulated_importance, final_imp = ga.importance(n_explain = 3)
    print(cumulated_importance)
    print(statistics.median([v1_imp[0] for v1_imp in cumulated_importance]))
    print(statistics.median([v1_imp[1] for v1_imp in cumulated_importance]))
    print(final_imp)

    # fig = plt.figure()
    # plt.plot(fitness)
    # plt.xlabel('Iteration')
    # plt.ylabel('Objective function value')
    # plt.show()





    









