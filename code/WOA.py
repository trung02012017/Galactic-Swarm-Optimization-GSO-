import numpy as np
import math
from copy import deepcopy


class WhaleOptimizationAlgorithm(object):

    def __init__(self, dimension, population_size, range0, range1, max_ep):
        self.dimension = dimension      # dimension size
        self.population_size = population_size
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i]**2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def init_population(self):
        return [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.population_size)]

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(self.dimension)])

    def get_prey(self, population):
        population_fitness = [self.get_fitness(whale) for whale in population]
        min_index = np.argmin(population_fitness)
        return population[min_index], np.amin(population_fitness)

    def shrink_encircling(self, current_whale, best_solution, C, A):
        D = np.abs(C*best_solution - current_whale)
        return best_solution - A*D

    def update_following_spiral(self, current_whale, best_solution, b, l):
        D = np.abs(best_solution - current_whale)
        return D*np.exp(b*l)*np.cos(2*np.pi*l) + best_solution

    def explore_new_prey(self, current_whale, C, A):
        random_whale = np.random.uniform(self.range0, self.range1, self.dimension)
        D = np.abs(C*random_whale - current_whale)
        return random_whale - A*D

    def evaluate_population(self, population):

        population = np.maximum(population, self.range0)
        for i in range(self.population_size):
            for j in range(self.dimension):
                if population[i, j] > self.range1:
                    population[i, j] = np.random.uniform(range1-1, range1, 1)

        return population

    def run(self, population):
        b = 1
        for epoch_i in range(self.max_ep):
            for i in range(self.population_size):
                current_whale = population[i]
                a = 1.5 - 1.5*epoch_i/self.max_ep
                # a = np.random.uniform(0, 2, 1)
                # a = 2*np.cos(epoch_i/self.max_ep)
                # a = math.log((4 - 3*epoch_i/(self.max_ep+1)), 2)

                a2 = -1 + epoch_i*((-1)/self.max_ep)
                r1 = np.random.random(1)
                r2 = np.random.random(1)
                A = 2*a*r1 - a
                C = 2*r2
                l = (a2 - 1)*np.random.random(1) + 1
                p = np.random.random(1)
                if p < 0.5:
                    if np.abs(A) < 1:
                        updated_whale = self.shrink_encircling(current_whale, self.best_solution, C, A)
                    else:
                        updated_whale = self.explore_new_prey(current_whale, C, A)
                else:
                    updated_whale = self.update_following_spiral(current_whale, self.best_solution, b, l)
                # updated_whale = self.update_Levy(updated_whale)
                population[i] = updated_whale

            population = self.evaluate_population(population)
            # self.best_solution, self.best_fitness = self.get_prey(population)
            new_best_solution, new_best_fitness = self.get_prey(population)
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
            print(self.best_fitness)
        print(self.best_solution)

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


if __name__ == '__main__':
    dimension = 50
    population_size = 100
    range0 = -10
    range1 = 10
    max_ep = 1000



    WOA = WhaleOptimizationAlgorithm(dimension, population_size, range0, range1, max_ep)
    population = WOA.init_population()
    WOA.run(population)

    # MWOA = ModifiedWOA(dimension, population_size, range0, range1, max_ep)
    # population = MWOA.init_population()
    # MWOA.run(population)