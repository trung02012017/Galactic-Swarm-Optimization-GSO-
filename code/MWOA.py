import numpy as np
from copy import deepcopy
import math

class ModifiedWOA(WhaleOptimizationAlgorithm):

    def __init__(self, dimension, population_size, range0, range1, max_ep):
        self.dimension = dimension      # dimension size
        self.population_size = population_size
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i]**2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def caculate_xichma(self, beta):
        up = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        down = (math.gamma((1+beta)/2)*beta*math.pow(2, (beta-1)/2))
        xich_ma_1 = math.pow(up/down, 1/beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def shrink_encircling_Levy(self, current_whale, best_solution, epoch_i, beta=2):
        xich_ma_1, xich_ma_2 = self.caculate_xichma(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01*a/(math.pow(np.abs(b), 1/beta))*(current_whale - best_solution)
        D = np.random.uniform(self.range0, self.range1, 1)
        levy = LB*D
        return current_whale + math.sqrt(epoch_i)*np.sign(np.random.random(1) - 0.5)*levy

    def run(self, population):
        b = 1
        for epoch_i in range(self.max_ep):
            for i in range(self.population_size):
                current_whale = population[i]
                a = 2 - 2*epoch_i/self.max_ep
                # a = 2*np.cos(epoch_i/self.max_ep)
                a2 = -1 + epoch_i*((-1)/self.max_ep)
                r1 = np.random.random(1)
                r2 = np.random.random(1)
                A = 2*a*r1 - a
                C = 2*r2
                l = (a2 - 1)*np.random.random(1) + 1
                p = np.random.random(1)
                if p < 0.5:
                    if np.abs(A) < 1:
                        updated_whale = self.shrink_encircling_Levy(current_whale, self.best_solution, epoch_i)
                    else:
                        updated_whale = self.explore_new_prey(current_whale, C, A)
                else:
                    updated_whale = self.update_following_spiral(current_whale, self.best_solution, b, l)
                population[i] = updated_whale

            population = self.evaluate_population(population)
            # self.best_solution, self.best_fitness = self.get_prey(population)
            new_best_solution, new_best_fitness = self.get_prey(population)
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
            print(self.best_fitness)
        print(self.best_solution)