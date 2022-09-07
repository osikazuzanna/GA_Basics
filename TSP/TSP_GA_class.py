from audioop import cross
from mimetypes import init
from matplotlib.cbook import sanitize_sequence
from numpy.random import rand, randint 
import numpy as np
import matplotlib.pyplot as plt
from numpy import min, sum, ptp, array
from random import sample
import operator
import pandas as pd
import random

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"






class GA:

    def __init__(self, population, pop_size, elite_size, mutation_rate, generations):
        self.population = population,
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        

    class Fitness:
        def __init__(self, route):
            self.route = route
            self.distance = 0
            self.fitness= 0.0
        
        def routeDistance(self):
            if self.distance ==0:
                pathDistance = 0
                for i in range(0, len(self.route)):
                    fromCity = self.route[i]
                    toCity = None
                    if i + 1 < len(self.route):
                        toCity = self.route[i + 1]
                    else:
                        toCity = self.route[0]
                    pathDistance += fromCity.distance(toCity)
                self.distance = pathDistance
            return self.distance
        
        def routeFitness(self):
            if self.fitness == 0:
                self.fitness = 1 / float(self.routeDistance())
            return self.fitness


    def createRoute(self):
        route = random.sample(cityList, len(self.population))
        return route

    def initialPopulation(self):
        population = []
        for i in range(0, self.pop_size):
            population.append(self.createRoute())
        return population

    def rankRoutes(self, population):
        fitnessResults = {}

        for i in range(0,len(population)):
            self.FitnessCalc = Fitness(population[i])
            fitnessResults[i] = self.FitnessCalc(population[i]).routeFitness()
        return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


    def selection(self, popRanked):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
        for i in range(0, self.elite_size):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - self.elite_size):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i,3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults

    def matingPool(self, population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    def breed(self, parent1, parent2):
        child = []
        childP1 = []
        childP2 = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min([geneA, geneB])
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(parent1[i])
            
        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child

    def breedPopulation(self, matingpool):
        children = []
        length = len(matingpool) - self.elite_size
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0,self.elite_size):
            children.append(matingpool[i])
        
        for i in range(0, length):
            child = self.breed(pool[i], pool[len(matingpool)-i-1])
            children.append(child)
        return children



    def mutate(self, individual):
        for swapped in range(len(individual)):
            if(random.random() < self.mutationRate):
                swapWith = int(random.random() * len(individual))
                
                city1 = individual[swapped]
                city2 = individual[swapWith]
                
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual


    def mutatePopulation(self, population):
        mutatedPop = []
        
        for ind in range(0, len(population)):
            mutatedInd = self.mutate(population[ind])
            mutatedPop.append(mutatedInd)
        return mutatedPop

    def nextGeneration(self, currentGen):
        popRanked = self.rankRoutes(currentGen)
        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool)
        nextGeneration = self.mutatePopulation(children)
        return nextGeneration
    

    def geneticAlgorithm(self):
        pop = self.initialPopulation()
        print("Initial distance: " + str(1 / self.rankRoutes(pop)[0][1]))
        
        for i in range(0, self.generations):
            pop = self.nextGeneration(pop)
        
        print("Final distance: " + str(1 / self.rankRoutes(pop)[0][1]))
        bestRouteIndex = self.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        return bestRoute


#--------------------------------------------------------------------------------------

if __name__ == '__main__':
    #parameters
    popSize=100 
    eliteSize=20
    mutationRate=0.01
    generations=500
    cityList = []
    for i in range(0,25):
        cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
    
    tsp = GA(cityList, popSize, eliteSize, mutationRate, generations)
    solution = tsp.geneticAlgorithm()


# geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)



# def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     progress = []
#     progress.append(1 / rankRoutes(pop)[0][1])
    
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#         progress.append(1 / rankRoutes(pop)[0][1])
    
#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.show()


# geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

