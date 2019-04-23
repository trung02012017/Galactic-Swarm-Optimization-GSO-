import numpy as np
from copy import deepcopy
import os
import json
from fitness_selector import Fitness_Selector


class ArtificialBeeColony(object):

    def __init__(self, fitness_function, dimension, population_size, population, range0, range1, max_ep):
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.population_size = population_size
        self.population = population
        self.range0 = range0
        self.range1 = range1
        self.abandonment_limit = population_size*dimension/5
        self.population_fitness = np.zeros(population_size)
        self.abandonment_counter = np.zeros(population_size)
        self.max_ep = max_ep
        self.gBest = np.random.uniform(range0, range1, dimension)

    def set_population_fitness(self):
        for i in range(self.population_size):
            self.population_fitness[i] = self.fitness_function(self.population[i])

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def recruited_bees_work(self):

        for i in range(self.population_size):
            k = np.random.randint(0, self.population_size-1)
            while k == i:
                k = np.random.randint(0, self.population_size - 1)

            bee = self.population[i]

            num_var = np.random.randint(0, self.dimension)
            permutation = np.random.permutation(self.dimension)
            vars_index = permutation[:num_var]

            phi = np.random.uniform(-1, 1, num_var)
            new_bee = deepcopy(bee)
            new_bee[vars_index] = bee[vars_index] + \
                                  phi * (bee[vars_index] - self.population[k][vars_index])
            # phi = np.random.uniform(-1, 1, self.dimension)
            # new_bee = bee + phi*(bee - population[k])
            new_bee = np.maximum(new_bee, self.range0)
            new_bee = np.minimum(new_bee, self.range1)

            new_bee_fitness = self.get_fitness(new_bee)

            if new_bee_fitness <= self.population_fitness[i]:
                self.population[i] = new_bee
                self.population_fitness[i] = new_bee_fitness
            else:
                self.abandonment_counter[i] += 1

    def roulette_wheel_selection(self, P):

        max = np.sum(P)
        pick = np.random.uniform(0, max)
        current = 0

        for i in range(self.population_size):
            current += P[i]
            if current > pick:
                return i

    def onlooker_bees_work(self):

        population_fitness = [self.get_fitness(self.population[i]) for i in range(self.population_size)]
        mean_fitness = np.mean(population_fitness)
        norm_fitness = population_fitness/mean_fitness*-1
        F = np.exp(norm_fitness)
        P = F/np.sum(F)

        population = deepcopy(self.population)
        for i in range(self.population_size):
            bee_index = self.roulette_wheel_selection(P)
            bee = deepcopy(population[bee_index])

            k = np.random.randint(0, self.population_size - 1)
            while k == i:
                k = np.random.randint(0, self.population_size - 1)

            num_var = np.random.randint(0, self.dimension)
            permutation = np.random.permutation(self.dimension)
            vars_index = permutation[:num_var]

            phi = np.random.uniform(-1, 1, num_var)
            new_bee = deepcopy(bee)
            new_bee[vars_index] = bee[vars_index] + phi * (bee[vars_index] - population[k][vars_index])

            # phi = np.random.uniform(-1, 1, self.dimension)
            # new_bee = bee + phi * (bee - population[k])

            new_bee = np.maximum(new_bee, self.range0)
            new_bee = np.minimum(new_bee, self.range1)

            new_bee_fitness = self.get_fitness(new_bee)
            current_bee_fitness = self.get_fitness(bee)

            if new_bee_fitness <= current_bee_fitness:
                population[i] = new_bee
            else:
                self.abandonment_counter[i] += 1
        self.population = deepcopy(population)
        self.set_population_fitness()

    def scouts_work(self):

        for i in range(self.population_size):
            if self.abandonment_counter[i] >= self.abandonment_limit:
                self.population[i] = np.random.uniform(self.range0, self.range1, self.dimension)
                self.abandonment_counter[i] = 0

    def find_best_solution(self):

        population_fitness = [self.get_fitness(self.population[i]) for i in range(self.population_size)]
        best_fitness = np.min(population_fitness)
        best_solution_index = np.where(population_fitness == best_fitness)[0][0]
        best_solution = self.population[best_solution_index]

        return best_solution, best_fitness

    def run(self):

        for i in range(self.max_ep):
            self.recruited_bees_work()
            self.onlooker_bees_work()
            self.scouts_work()
            best_solution, best_fitness = self.find_best_solution()
            if best_fitness <= self.get_fitness(self.gBest):
                self.gBest = deepcopy(best_solution)
            print("iter: {} and best_fitness: {}".format(i, self.get_fitness(self.gBest)))


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup', 'params_demo.json')

    with open(params_path, 'r') as f:
        data = json.load(f)

    parameter_set = data["parameters"]

    fitness_function_names = parameter_set['function']
    dimensions = parameter_set["dimension"]
    range0_s = parameter_set["range0"]
    range1_s = parameter_set["range1"]

    population_sizes = parameter_set["ABC"]['population_size']
    abandonment_limits = parameter_set["ABC"]['abandonment_limit']
    max_eps = parameter_set["ABC"]['max_ep']

    combinations = []

    for fitness_function_name in fitness_function_names:
        if fitness_function_name == 'f18' or fitness_function_name == 'f19':
            continue
        for dimension in dimensions:
            for range0 in range0_s:
                for range1 in range1_s:
                    if round(range0 + range1) != 0:
                        continue
                    for population_size in population_sizes:
                        for abandonment_limit in abandonment_limits:
                            for max_ep in max_eps:
                                function_evaluation = population_size * max_ep
                                combination = [fitness_function_name, dimension, range0, range1,
                                               population_size, abandonment_limit, max_ep,
                                               function_evaluation]
                                # if 30000 >= function_evaluation >= 25000:
                                combinations.append(combination)

    for combination in combinations:
        fitness_function_name = combination[0]
        dimension = combination[1]
        range0 = combination[2]
        range1 = combination[3]
        population_size = combination[4]
        abandonment_limit = combination [5]
        max_ep = combination[6]

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]

        ABC = ArtificialBeeColony(fitness_function, dimension, population_size, population, range0, range1, max_ep)

        ABC.run()