import numpy as np
from copy import deepcopy
import math
import time
import pandas as pd
import os
import json
from fitness_selector import Fitness_Selector

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
        random_index = np.random.randint(0, self.population_size)
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
        LB = 0.01*a/(math.pow(np.abs(b), 1/beta))*(C*current_whale - best_solution)
        D = np.random.uniform(self.range0, self.range1, 1)
        levy = LB*D
        return (current_whale + math.sqrt(epoch_i + 1)*np.sign(np.random.random(1) - 0.5))*levy

    def crossover(self, population):
        partner_index = np.random.randint(0, self.population_size)
        partner = population[partner_index]
        # partner = np.random.uniform(self.range0, self.range1, self.dimension)

        start_point = np.random.randint(0, self.dimension/2)
        new_whale = np.zeros(self.dimension)

        index1 = start_point
        index2 = int(start_point+self.dimension/2)
        index3 = int(self.dimension)

        new_whale[0:index1] = self.best_solution[0:index1]
        new_whale[index1:index2] = partner[index1:index2]
        new_whale[index2:index3] = self.best_solution[index2:index3]

        return new_whale

    def run(self):
        b = 1
        gBest_collection = np.zeros(self.max_ep)
        start_time = time.clock()
        for epoch_i in range(self.max_ep):
            for i in range(self.population_size):
                current_whale = self.population[i]
                a = 2 - 2*epoch_i/self.max_ep
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
            gBest_collection[epoch_i] = self.get_fitness(self.best_solution)
        total_time = time.clock() - start_time
        return self.get_fitness(self.best_solution), gBest_collection, total_time


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

    population_sizes = parameter_set["MWOA"]['population_size']
    max_eps = parameter_set["MWOA"]['max_ep']

    stability_number = 20
    combinations = []

    for fitness_function_name in fitness_function_names:
        if fitness_function_name == 'f18' or fitness_function_name == 'f19':
            continue
        for dimension in dimensions:
            if fitness_function_name in ["f1", "f3", "f4", "f6", "f8", "f11", "f22"]:
                range0, range1 = -100, 100
            if fitness_function_name in ["f2", "f10", "f12", "f13", "f21"]:
                range0, range1 = -10, 10
            if fitness_function_name in ["f9", "f25"]:
                range0, range1 = -1, 1
            if fitness_function_name in ["f5"]:
                range0, range1 = -30, 30
            if fitness_function_name in ["f7"]:
                range0, range1 = -1.28, 1.28
            if fitness_function_name in ["f14"]:
                range0, range1 = -500, 500
            if fitness_function_name in ["f15"]:
                range0, range1 = -5.12, 5.12
            if fitness_function_name in ["f16"]:
                range0, range1 = -32, 32
            if fitness_function_name in ["f17"]:
                range0, range1 = -60, 60
            if fitness_function_name in ["f23"]:
                range0, range1 = -5, 5
            if fitness_function_name in ["f24"]:
                range0, range1 = -15, 15
            for population_size in population_sizes:
                for max_ep in max_eps:
                    function_evaluation = population_size * max_ep
                    combination = [fitness_function_name, dimension, range0, range1,
                                   population_size, max_ep, function_evaluation]
                    combinations.append(combination)
    print(len(combinations))

    def save_result(combination, all_gbests, gBest_fitness, total_time):

        fitness_function_name = combination[0]
        path = '../results/' + str(fitness_function_name) + '/MWOA/'
        path1 = path + 'error_MWOA' + str(combination) + '.csv'
        path2 = path + 'models_log.csv'
        combination = [combination]
        error = {
            'epoch': range(1, 1 + all_gbests.shape[0]),
            'gBest_fitness': all_gbests,
        }

        model_log = {
            'combination': combination,
            'total_time': round(total_time, 2),
            'gBest_fitness': gBest_fitness,
        }

        df_error = pd.DataFrame(error)
        if not os.path.exists(path1):
            columns = ['epoch', 'gBest_fitness']
            df_error.columns = columns
            df_error.to_csv(path1, index=False, columns=columns)
        else:
            with open(path1, 'a') as csv_file:
                df_error.to_csv(csv_file, mode='a', header=False, index=False)

        df_models_log = pd.DataFrame(model_log)
        if not os.path.exists(path2):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, population_size, max_ep, '
                       'function evaluation]',
                       'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, fitness_function_name, gBest_fitness, total_time):
        path = '../results/' + str(fitness_function_name) + '/MWOA/'
        path3 = path + 'stability_igso.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }

        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, population_size, max_ep, '
                       'function evaluation]',
                       'gBest_fitness', 'total_time']
            df_stability.columns = columns
            df_stability.to_csv(path3, index=False, columns=columns)
        else:
            with open(path3, 'a') as csv_file:
                df_stability.to_csv(csv_file, mode='a', header=False, index=False)


    # fitness_selector = Fitness_Selector()
    # fitness_function = fitness_selector.chose_function('f1')
    #
    # GSO = GalacticSwarmOptimization(fitness_function, 50, -10, 10, 15, 5, 10, 300, 5, 2.5, 2.5)
    # subswarm_collection = GSO.init_population()
    # fitness_gBest, gBest_fitness_collection, total_time = GSO.run(subswarm_collection)
    # print(gBest_fitness_collection)

    model_name = "MWOA"

    for combination in combinations:
        fitness_function_name = combination[0]
        dimension = combination[1]
        range0 = combination[2]
        range1 = combination[3]
        population_size = combination[4]
        max_ep = combination[5]

        path = "../results/" + fitness_function_name
        if not os.path.exists(path):
            os.mkdir(path)

        path = "../results/" + fitness_function_name + "/" + model_name
        if not os.path.exists(path):
            os.mkdir(path)

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
        MWOA = ModifiedWOA(fitness_function, dimension, population_size, population, range0, range1, max_ep)
        fitness_gBest, gBest_fitness_collection, total_time = MWOA.run()
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time: {}'.format(combination, fitness_gBest, total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            population_i = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
            MWOA_i = ModifiedWOA(fitness_function, dimension, population_size, population_i, range0, range1, max_ep)
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = MWOA_i.run()
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_function_name, fitness_gBest, total_time)