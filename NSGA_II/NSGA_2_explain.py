from lib2to3.pgen2.token import LPAR
from random import random
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn
#Problem Class

class MOO_Problem():
    '''
    Class defines the dfferent constraint function that are needed to be minimized.
    Param:
        d: Dimension of input vector for the constraint functions
        lower: (mx1) numpy array giving lower bound of the m constraints. Default = -1
        upper: (mx1) numpy array giving upper bound of the m constraints. Default = 1
        *args: m constraints in the form of python functions capable of running on (Nxd) numpy arrays
    Usage: MOO_Problem(func1, func2, func3, ...)
    '''
    def __init__(self, *args, d=None,lower=None, upper=None):
        if d is None:
            print("Missing required parameter 'd'")
            sys.exit(-1)
        self.d = d
        self.m = len(args)
        if self.m < 1:
            print("Enter atleast one constraint function")
            sys.exit(-1)
        self.constraints = list(args)
        self.lower = lower if lower is not None else -np.ones((1, self.d))
        self.upper = upper if upper is not None else np.ones((1, self.d))
    
    def evaluate(self, x):
        '''
        Evaluate the constriants on multiple possible vectors
        Param: x: (N, m) numpy matrix where N is number of vectors and m is number of constraints
        Return: (N, m) numpy array of evaluated results
        '''
        N = x.shape[0]
        obj = np.empty((N, self.m))
        for i in range(self.m):
            obj[:, i] = np.squeeze(self.constraints[i](x))
        return obj



## DTLZ 2 Problem

m = 3
d = 4
lb = np.array([[0]*d])
ub = np.array([[1]*d])

def g(x):
    return np.sum(np.square(x[:, 2:]-0.5), axis=1)

def f1(x):
    return -((1+g(x))*np.cos(np.pi/2*x[:, 0])*np.cos(np.pi/2*x[:, 1]))

def f2(x):
    return -((1+g(x))*np.sin(np.pi/2*x[:, 1])*np.cos(np.pi/2*x[:, 0]))

def f3(x):
    return -((1+g(x))*np.sin(np.pi/2*x[:, 0]))

moo_problem = MOO_Problem(f1, f2, f3, d=d, lower=lb, upper=ub)


# nsgaii = NSGAII(moo_problem, pop_size=100, n_iter=100)
# ax = nsgaii.visualize()
# ax.view_init(elev=-145., azim=45)
# ax.set_title('DTLZ 2 Problem')
# ax.set_xlabel('f1(x)')
# ax.set_ylabel('f2(x)')
# ax.set_zlabel('f3(x)')
# plt.show()





#NSGA2


N = 100 #population size
p_m = 1.0 #mutation probability
p_c = 1.0  #crossover probability
max_iter = 100 #maximum number of operations allowed
max_evals = 100*500 #maximum number of function evaluations (per individual 500)

#random population:

random_pop = np.random.random((N, d)) * (lb - ub) + lb
pop = [random_pop, moo_problem.evaluate(random_pop)]


def non_dominated_sort(n_sort = None):
    '''
    Performs non dominated sort of the population in this iteration
    Param: n_sort: maximum number of individuals to sort
    Return:
        front_ids: (Nx1) vctor where front_id[i] is the front number of pop[i]
        max_front: Number of fronts calculated
    '''
    # Initialization
    pop_cost = pop[1]
    N = pop_cost.shape[0]
    _, loc = np.unique(pop_cost[:,0], return_inverse=True)
    if n_sort is None: n_sort = len(loc)

    sorted_cost = pop_cost[pop_cost[:,0].argsort(), :]
    front_id = np.inf*np.ones(N)
    max_front = 0
    #Non dominated sort
    while np.sum(front_id < np.inf) < n_sort:      # while individuals left without front_id
        max_front += 1
        for i in np.where(front_id==np.inf)[0]:
#                 if np.sum(front_id < np.inf) >= n_sort: break
            dominated = False
            for j in range(i, 0, -1):
                if front_id[j-1] == max_front:
                    m=2
                    while(m<=moo_problem.m) and (sorted_cost[i, m-1] >= sorted_cost[j-1, m-1]):
                        m += 1
                    dominated = m > moo_problem.m
                    if dominated or moo_problem.m==2:
                        break
            if not dominated:
                front_id[i] = max_front
    return front_id[loc], max_front


def crowding_distance(front_id):
    '''
    Calculate the crowding distance for each pareto front
    Param: front_id: (Nx1) numpy array where front_i[i] is the front number for pop[i]
    Return: crowd_dis: (Nx1) numpy array of crowding distances
    '''
    N = pop[0].shape[0]
    pop_cost = pop[1]
    crowd_dis = np.zeros(N)
    fronts = np.unique(front_id)
    fronts = fronts[fronts!=np.inf]
    
    for f in range(len(fronts)):
        front = np.where(front_id==f+1)[0]
        fmax = np.max(pop_cost[front, :], axis=0)
        fmin = np.min(pop_cost[front, :], axis=0)
        for i in range(moo_problem.m):
            rank = np.argsort(pop_cost[front, i])
            crowd_dis[front[rank[0]]] = np.inf
            crowd_dis[front[rank[-1]]] = np.inf
            for j in range(1, len(front)-1):
                crowd_dis[front[rank[j]]] = crowd_dis[front[rank[j]]] + \
                        (pop_cost[front[rank[j+1]], i] - pop_cost[front[rank[j-1]], i]) / (fmax[i]-fmin[i])
    return crowd_dis





def tournament(fit, K=2):
    '''
    Perform crowded tournament
    Param:
        K: Number of parameters to be considered for fitness
        fit: (N, K) matrix of fitness values, where the higher column means higher preference
    Return: indices of individuals who won the tournament
    '''
    n_total = len(fit)
    a = np.random.randint(n_total, size=N)
    b = np.random.randint(n_total, size=(N, K))
    for i in range(N):
        for j in range(K):
            for r in range(fit[0, :].size):
                if fit[b[i, j], r] < fit[a[i], r]:
                    a[i] = b[i,j]
    return a





def evolve(parents, boundary=None):
    ''' Creates the offspring from parents through crossover + mutation
    Param:
        parents: (Nxm) matrix of parents to participate in mutation
        boundary: (lower, upper) bounds of the constraints. Defualt value taken from moo problem.
    '''
    dis_c, dis_m = 20, 20
    parents = parents[:(len(parents)//2)*2, :]
    (n, d) = parents.shape
    parent_1, parent_2 = parents[:n//2, :], parents[n//2:, :]
    
    ## CROSSOVER
    beta = np.empty((n//2, d))
    mu = np.random.random((n//2, d))
    beta[mu <= 0.5]= np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
    beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
    beta = beta * ((-1)** np.random.randint(2, size=(n // 2, d)))
    beta[np.random.random((n // 2, d)) < 0.5] = 1
    beta[np.tile(np.random.random((n // 2, 1)) > p_c, (1, d))] = 1
    # beta=1 means no crossover
    
    offspring = np.vstack(((parent_1 + parent_2) / 2 + beta * (parent_1 - parent_2) / 2,
                                (parent_1 + parent_2) / 2 - beta * (parent_1 - parent_2) / 2))

    offspring.astype('float64')

    # MUTATION
    site = np.random.random((n, d)) < p_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    if boundary is None:
        lower, upper = np.tile(moo_problem.lower, (n, 1)), np.tile(moo_problem.upper, (n,1))
    else:
        lower, upper = boundary
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                            (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                        1. / (dis_m + 1)) - 1.)
    temp = ~temp
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp])* \
                            (1. - np.power(np.abs(2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(np.abs(1. - norm), dis_m + 1.)), 1. / (dis_m + 1.)))
    # offspring = np.maximum(np.minimum(offspring, upper), lower)
    return offspring


def selection():
    '''Performs the environment selection based on the front_ids and crowding_distance
    '''
    front_id, max_front = non_dominated_sort(n_sort=N)
    next_label = np.zeros(front_id.shape[0], dtype=bool)
    next_label[front_id<max_front] = True
    crowd_dis = crowding_distance(front_id)
    last = np.where(front_id==max_front)[0]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[:(N - int(np.sum(next_label)))]
    next_label[last[delta_n]] = True
    index = np.where(next_label)[0]
    pop = [pop[0][index, :], pop[1][index, :]]
    return front_id[index], crowd_dis[index], index

def run():
    random_pop = np.random.random((N, d)) * (lb - ub) + lb
    pop = [random_pop, moo_problem.evaluate(random_pop)]
    front_id, max_front = non_dominated_sort()
    crowd_dis = crowding_distance(front_id)
    surrogate = pd.DataFrame()
    surrogate_performance = []
    eval_left, n_iter = max_evals, 0
    while eval_left >= 0 and n_iter <= max_iter:
        fit = np.vstack((front_id, crowd_dis)).T
        mating_pool = tournament(fit)
        parent = [pop[0][mating_pool, :], pop[1][mating_pool, :]]
        offspring = evolve(parent[0])
        offspring_cost = moo_problem.evaluate(offspring)
        pop = [np.vstack((pop[0], offspring)), np.vstack((pop[1], offspring_cost))]
        front_id, max_front = non_dominated_sort(n_sort=N)
        next_label = np.zeros(front_id.shape[0], dtype=bool)
        next_label[front_id<max_front] = True
        crowd_dis = crowding_distance(front_id)
        last = np.where(front_id==max_front)[0]
        rank = np.argsort(-crowd_dis[last])
        delta_n = rank[:(N - int(np.sum(next_label)))]
        next_label[last[delta_n]] = True
        index = np.where(next_label)[0]
        pop = [pop[0][index, :], pop[1][index, :]]

    



        if n_iter == 0 :
            surrogate_dataset_pareto = pop[0][front_id[index] == 1, :]
            n_pareto = len(surrogate_dataset_pareto)
            surrogate_dataset_pareto = np.concatenate([surrogate_dataset_pareto, np.array([[1]]*n_pareto, dtype = 'int64')], axis = 1)
            surrogate_dataset_nonpareto = pop[0][np.random.choice(np.where(front_id[index] != 1)[0], n_pareto), : ]
            surrogate_dataset_nonpareto = np.concatenate([surrogate_dataset_nonpareto, np.array([[0]]*n_pareto, dtype = 'int64')], axis = 1)
            surrogate = surrogate.append([pd.DataFrame(surrogate_dataset_pareto), pd.DataFrame(surrogate_dataset_nonpareto)])

        elif n_iter % 5 != 0:
            surrogate_dataset_pareto = pop[0][front_id[index] == 1, :]
            n_pareto = len(surrogate_dataset_pareto)
            surrogate_dataset_pareto = np.concatenate([surrogate_dataset_pareto, np.array([[1]]*n_pareto, dtype = 'int64')], axis = 1)
            surrogate_dataset_nonpareto = pop[0][np.random.choice(np.where(front_id[index] != 1)[0], n_pareto), : ]
            surrogate_dataset_nonpareto = np.concatenate([surrogate_dataset_nonpareto, np.array([[0]]*n_pareto, dtype = 'int64')], axis = 1)
            surrogate = surrogate.append([pd.DataFrame(surrogate_dataset_pareto), pd.DataFrame(surrogate_dataset_nonpareto)])

        else:
            surrogate_dataset_pareto = pop[0][front_id[index] == 1, :]
            n_pareto = len(surrogate_dataset_pareto)
            surrogate_dataset_pareto = np.concatenate([surrogate_dataset_pareto, np.array([[1]]*n_pareto)], axis = 1)
            surrogate_dataset_nonpareto = pop[0][np.random.choice(np.where(front_id[index] != 1)[0], n_pareto), : ]
            surrogate_dataset_nonpareto = np.concatenate([surrogate_dataset_nonpareto, np.array([[0]]*n_pareto)], axis = 1)
            surrogate = surrogate.append([pd.DataFrame(surrogate_dataset_pareto), pd.DataFrame(surrogate_dataset_nonpareto)])
            train, test = sklearn.model_selection.train_test_split(surrogate, test_size=0.2)
            clf = RandomForestClassifier(max_depth=2, random_state=0).fit(train.iloc[:,:train.shape[1] - 1], train.iloc[:,train.shape[1] - 1])
            performance = sklearn.metrics.roc_auc_score(test.iloc[:,test.shape[1] - 1], clf.predict_proba(test.iloc[:,:test.shape[1] - 1])[:,1])
            surrogate_performance.append(performance)
            surrogate = pd.DataFrame()

        eval_left -= N
        n_iter +=1
            
    print("Num iteration: {}, Num. function evaluations: {}".format(n_iter, max_evals-eval_left))
    return pop, front_id[index], crowd_dis[index], index, surrogate, clf, surrogate_performance

pop_, fronts, crowd, ind, surr, rfc, perf = run()

print(pop_)
print(fronts)
# train, test = sklearn.model_selection.train_test_split(surr, test_size=0.2)
# clf = RandomForestClassifier(max_depth=2, random_state=0).fit(train.iloc[:,:train.shape[1] - 1], train.iloc[:,train.shape[1] - 1])
# performance = sklearn.metrics.roc_auc_score(test.iloc[:,test.shape[1] - 1], clf.predict_proba(test.iloc[:,:test.shape[1] - 1])[:,1])
# print(performance)



