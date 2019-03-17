import numpy as np
import math
from copy import deepcopy


class WhaleOptimizationAlgorithm(object):

    def __init__(self, dimension, population_size, population, range0, range1, max_ep):
        self.dimension = dimension      # dimension size
        self.population_size = population_size
        self.population = population
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i]**2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(self.dimension)])
    def set_best_solution(self, best_solution):
        self.best_solution = best_solution

    def get_prey(self):
        population_fitness = [self.get_fitness(whale) for whale in self.population]
        min_index = np.argmin(population_fitness)
        return self.population[min_index], np.amin(population_fitness)

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
                    population[i, j] = np.random.uniform(self.range1-1, self.range1, 1)

        return population

    def run(self):
        b = 1
        for epoch_i in range(self.max_ep):
            for i in range(self.population_size):
                current_whale = self.population[i]
                a = 1.5 - 1.5*epoch_i/self.max_ep
                # a = np.random.uniform(0, 2, 1)
                # a = 2*np.cos(epoch_i/self.max_ep)
                # a = math.log((4 - 3*epoch_i/(self.max_ep+1)), 2)
                # a = 2 * np.cos(epoch_i / self.max_ep)
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
                self.population[i] = updated_whale

            self.population = self.evaluate_population(self.population)
            # self.best_solution, self.best_fitness = deepcopy(self.get_prey())
            new_best_solution, new_best_fitness = self.get_prey()
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
            print(self.best_fitness)
        print(self.best_solution)
        return self.best_solution, self.get_fitness(self.best_solution)


if __name__ == '__main__':
    dimension = 50
    population_size = 100
    range0 = -10
    range1 = 10
    max_ep = 100

    population = ([np.random.uniform(range0, range1, dimension) for _ in range(population_size)])
    WOA = WhaleOptimizationAlgorithm(dimension, population_size, population, range0, range1, max_ep)
    WOA.run()