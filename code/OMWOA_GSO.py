import numpy as np
from copy import deepcopy
import math

class ModifiedWOA(object):

    def __init__(self, fitness_function, dimension, population_size, population, range0, range1, max_ep):
        self.fitness_function = fitness_function
        self.dimension = dimension  # dimension size
        self.population_size = population_size
        self.population = population
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i] ** 2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def init_population(self):
        return ([np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.population_size)])

    def get_fitness(self, particle):
        return self.fitness_function(particle)

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
        random_index = np.random.randint(0, self.population_size, size=1)
        random_whale = self.population[random_index]
        D = np.abs(C*random_whale - current_whale)
        return random_whale - A*D

    def evaluate_population(self, population):

        population = np.maximum(population, self.range0)
        for i in range(self.population_size):
            for j in range(self.dimension):
                if population[i, j] > self.range1:
                    population[i, j] = np.random.uniform(self.range1-1, self.range1, 1)

        return population

    def caculate_xichma(self, beta):
        up = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        down = (math.gamma((1+beta)/2)*beta*math.pow(2, (beta-1)/2))
        xich_ma_1 = math.pow(up/down, 1/beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def shrink_encircling_Levy(self, current_whale, best_solution, epoch_i, C,  beta=1):
        xich_ma_1, xich_ma_2 = self.caculate_xichma(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01*a/(math.pow(np.abs(b), 1/beta))*(-1*current_whale - C*best_solution)
        D = np.random.uniform(self.range0, self.range1)
        levy = LB*D
        return (current_whale + 1/math.sqrt(epoch_i + 1)*np.sign(np.random.random(1) - 0.5))*levy

    def quadratic_interpolation(self, population):
        partner_index = np.random.randint(0, self.population_size, 2)
        partner1 = population[partner_index[0]]
        partner2 = population[partner_index[1]]
        # partner = np.random.uniform(self.range0, self.range1, self.dimension)

        new_whale = 0.5*((partner1**2 - partner2**2)*self.get_fitness(self.best_solution) +
                         (partner2**2 - self.best_solution**2)*self.get_fitness(partner1) +
                         (self.best_solution**2 - partner1**2)*self.get_fitness(partner2)) / \
                         ((partner1 - partner2) * self.get_fitness(self.best_solution) +
                          (partner2 - self.best_solution) * self.get_fitness(partner1) +
                          (self.best_solution - partner1) * self.get_fitness(partner2))

        return new_whale

    def run(self):
        b = 1
        results = np.zeros(self.max_ep)
        for epoch_i in range(self.max_ep):
            for i in range(self.population_size):
                current_whale = self.population[i]
                # a = 2 - 2*epoch_i/self.max_ep
                # a = np.random.uniform(0, 2, 1)
                # a = 2*np.cos(epoch_i/self.max_ep)
                # a = math.log((4 - 3*epoch_i/(self.max_ep+1)), 2)
                a = 2 * np.cos(epoch_i / self.max_ep)
                a2 = -1 + epoch_i*((-1)/self.max_ep)
                r1 = np.random.random(1)
                r2 = np.random.random(1)
                A = 2*a*r1 - a
                C = 2*r2
                l = (a2 - 1)*np.random.random(1) + 1
                p = np.random.random(1)
                p1 = np.random.random(1)
                if p < 0.5:
                    if np.abs(A) < 1:
                        updated_whale = self.shrink_encircling_Levy(current_whale, self.best_solution, epoch_i, C)
                    else:
                        updated_whale = self.explore_new_prey(current_whale, C, A)
                else:
                    updated_whale = self.update_following_spiral(current_whale, self.best_solution, b, l)
                self.population[i] = updated_whale

            self.population = self.evaluate_population(self.population)
            # self.best_solution, self.best_fitness = self.get_prey(population)
            new_best_solution, new_best_fitness = self.get_prey()
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
            results[epoch_i] += self.best_fitness
        return self.best_solution, self.get_fitness(self.best_solution), results