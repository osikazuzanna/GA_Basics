
import matplotlib.pyplot as plt


#----------------------------------------------------------
#Initial population
from Binary_GA_explain import Binary_GA


# if __name__ == '__main__':
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
 
fig = plt.figure()
plt.plot(fitness)
plt.xlabel('Iteration')
plt.ylabel('Objective function value')
plt.show()