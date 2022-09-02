import numpy as np
from numpy.random import rand, randint 
import random
import pandas as pd
import statistics

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import sklearn
# #Parameters

# bounds = [[-10, 10], [-10, 10]]
# iteration = 200
# bits_per_var = 20
# n_var = 2
# pop_size = 20
# crossover_rate = 0.7
# mutation_rate = 0.3



# def init_pop(pop_size, bits_per_var, n_var):
#     pop = np.zeros([pop_size, bits_per_var*2], dtype=int)
#     for ind in range(pop_size):
#         for var in range(n_var):
#             for bit in range(bits_per_var):
#                 if (rand() <= 0.5):
#                     pop[ind][(var+1)*bit] = 1
#                 else:
#                     pop[ind][(var+1)*bit] = 0
#     return pop

# pop = init_pop(pop_size, bits_per_var, n_var)

# offspring = np.empty((0, 40))

# for i in range(int(len(pop)/2)):
#     p1 = pop[2*i-1].copy()
#     p2 = pop[2*i].copy()

#     if rand() < crossover_rate:
#         cutting_point = randint(1, len(p1)-1, size = 2)
#         while cutting_point[0] == cutting_point[1]:
#             cutting_point = randint(1, len(p1) - 1, size = 2)
#         cutting_point = sorted(cutting_point)
#         c1 = np.array(list(p1[:cutting_point[0]]) + list(p2[cutting_point[0]:cutting_point[1]]) + list(p1[cutting_point[1]:]))
#         c2 = np.array(list(p2[:cutting_point[0]]) + list(p1[cutting_point[0]:cutting_point[1]]) + list(p2[cutting_point[1]:]))
#         offspring = np.vstack([offspring, c1, c2])
#     else:
#         offspring = np.vstack([offspring, p1, p2])



# print(offspring.shape)

X, y = make_classification(n_samples=1000, n_features=4,
                         n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

DataFrame = pd.DataFrame(pd.DataFrame(X))
DataFrame['class'] = y
print(DataFrame.shape)

train, test = sklearn.model_selection.train_test_split(DataFrame, test_size=0.2)


clf = RandomForestClassifier(max_depth=2, random_state=0).fit(train.iloc[:,:train.shape[1] - 1], train.iloc[:,train.shape[1] - 1])

print(roc_auc_score(test.iloc[:,test.shape[1] - 1], clf.predict_proba(test.iloc[:,:test.shape[1] - 1])[:,1]))
