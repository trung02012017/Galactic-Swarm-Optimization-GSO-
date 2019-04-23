import numpy as np
from copy import deepcopy
import pandas as pd
import os
import time
import json
from fitness_selector import Fitness_Selector

class ParticleSwarmOptimization(object):

    def __init__(self, fitness_function, varsize, swarmsize, position, epochs, range0, range1, c1, c2):
        self.fitness_function = fitness_function
        self.varsize = varsize
        self.swarmsize = swarmsize
        self.epochs = epochs
        self.range0 = range0
        self.range1 = range1
        self.c1 = c1
        self.c2 = c2
        self.position = position
        self.velocity = np.zeros((swarmsize, varsize))
        self.pBest = position
        self.gBest = np.random.uniform(range0, range1, varsize)
        self.temp = self.gBest

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def set_gBest(self, gBest):
        self.gBest = gBest

    def run(self):

        v_max = 10
        w_max = 0.9
        w_min = 0.4

        gBest_collection = np.zeros(self.epochs)
        start_time = time.clock()
        for iter in range(self.epochs):
            w = (self.epochs - iter) / self.epochs * (w_max - w_min) + w_min
            # w = 1 - iter/(self.epochs + 1)
            for i in range(self.swarmsize):
                r1 = np.random.random()
                r2 = np.random.random()
                position_i = self.position[i]
                new_velocity_i = w*self.velocity[i] \
                                 + self.c1*r1*(self.pBest[i] - position_i) \
                                 + self.c2*r2*(self.gBest - position_i)
                new_velocity_i = np.maximum(new_velocity_i, -0.1 * v_max)
                new_velocity_i = np.minimum(new_velocity_i, 0.1 * v_max)
                new_position_i = position_i + new_velocity_i

                new_position_i = np.maximum(new_position_i, self.range0)
                for j in range(self.varsize):
                    if new_position_i[j] > self.range1:
                        new_position_i[j] = np.random.uniform(self.range1 - 1, self.range1, 1)

                fitness_new_pos_i = self.get_fitness(new_position_i)
                fitness_pBest = self.get_fitness(self.pBest[i])
                fitness_gBest = self.get_fitness(self.gBest)
                if fitness_new_pos_i < fitness_pBest:
                    self.pBest[i] = deepcopy(new_position_i)
                    if fitness_new_pos_i < fitness_gBest:
                        self.gBest = deepcopy(new_position_i)
                self.velocity[i] = new_velocity_i
                self.position[i] = new_position_i
            gBest_collection[iter] += self.get_fitness(self.gBest)
            # print(self.get_fitness(self.gBest))
        total_time = round(time.clock() - start_time, 2)
        # print(total_time)
        return self.get_fitness(self.gBest), gBest_collection, total_time


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup', 'params.json')

    with open(params_path, 'r') as f:
        data = json.load(f)

    parameter_set = data["parameters"]

    fitness_function_names = parameter_set['function']
    dimensions = parameter_set["dimension"]
    range0_s = parameter_set["range0"]
    range1_s = parameter_set["range1"]

    population_sizes = parameter_set["PSO"]['population_size']
    max_eps = parameter_set["PSO"]['max_ep']
    c1_s = parameter_set["PSO"]['c1']
    c2_s = parameter_set["PSO"]['c1']

    stability_number = 20
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
                        for max_ep in max_eps:
                            for c1 in c1_s:
                                for c2 in c2_s:
                                    function_evaluation = population_size * max_ep
                                    combination = [fitness_function_name, dimension, range0, range1,
                                                   population_size, max_ep, c1, c2, function_evaluation]
                                    if 30000 >= function_evaluation >= 25000:
                                        combinations.append(combination)
    print(len(combinations))

    def save_result(combination, all_gbests, gBest_fitness, total_time):

        fitness_function_name = combination[0]
        path = '../results/' + str(fitness_function_name) + '/PSO/'
        path1 = path + 'error_PSO' + str(combination) + '.csv'
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
                       'c1, c2, function evaluation]',
                       'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, fitness_function_name, gBest_fitness, total_time):
        path = '../results/' + str(fitness_function_name) + '/PSO/'
        path3 = path + 'stability_pso.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }

        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, population_size, max_ep, '
                       'c1, c2, function evaluation]',
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

    model_name = "PSO"

    for combination in combinations:
        fitness_function_name = combination[0]
        dimension = combination[1]
        range0 = combination[2]
        range1 = combination[3]
        population_size = combination[4]
        max_ep = combination[5]
        c1 = combination[6]
        c2 = combination[7]

        path = "../results/" + fitness_function_name + "/" + model_name
        if not os.path.exists(path):
            os.mkdir(path)

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
        PSO = ParticleSwarmOptimization(fitness_function, dimension, population_size, population, max_ep, range0, range1, c1, c2)
        fitness_gBest, gBest_fitness_collection, total_time = PSO.run()
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time: {}'.format(combination, fitness_gBest, total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            population_i = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
            PSO_i = ParticleSwarmOptimization(fitness_function, dimension, population_size, population_i,
                                              max_ep, range0, range1, c1, c2)
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = PSO_i.run()
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_function_name, fitness_gBest, total_time)


