import numpy as np
import time
import pandas as pd
import os.path
from copy import deepcopy
import json
from PSO_GSO import PSO
from GA_GSO import GenericAlgorithm
from fitness_selector import Fitness_Selector


class GalacticSwarmOptimization(object):

    def __init__(self, fitness_function, dimension, range0, range1, m, n, l1, l2, ep_max, crossover_rate, mutation_rate,
                 c3, c4):
        self.fitness_function = fitness_function
        self.dimension = dimension      # dimension size
        self.range0 = range0            # lower boundary of the value for each dimension
        self.range1 = range1            # upper boundary of the value for each dimension
        self.m = m                      # the number of subswarms that the population is divided into
        self.n = n                      # the number of particles per each subswarm ( so population size = m x n )
        self.l1 = l1                    # the number of epochs of PSO in phase 1
        self.l2 = l2                    # the number of epochs of PSO in phase 2
        self.ep_max = ep_max            # the number of epochs of PSO the whole system
        self.crossover_rate = crossover_rate                    # c1, c2 is parameters for the formula in phase 1
        self.mutation_rate = mutation_rate                    # c3, c4 is parameters for the formula in phase 2
        self.c3 = c3
        self.c4 = c4
        self.gBest = None

    def init_population(self):  # initialize population by setting up randomly each subswarm
        subswarm_collection = []
        for i in range(self.m):
            subswarm_i = [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.n)]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def run_phase_1(self, subswarm_collection, GA_list=None): # run PSO in subswarms independently
        gBest_collection = np.zeros((self.m, self.dimension))   # set of gBests of all subswarms after running PSO
        gBest_fitness_collection = np.zeros(self.m)             # set of all gBest fitness (just for showing the result)
        if GA_list is None:   # at epoch 1, PSO objects are created, at the end of each epoch,
                                # the states of each subswarm is saved and continued in next epoch
            GA_list = []
            for i in range(self.m):
                subswarm_i = subswarm_collection[i]
                GA_i = GenericAlgorithm(self.fitness_function, self.dimension, self.n, subswarm_i, self.crossover_rate,
                                        self.mutation_rate, self.l1)
                gBest_collection[i], gBest_fitness_collection[i] = GA_i.run()
                GA_list.append(GA_i)
                # print("gBest of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        else:
            for i in range(self.m): # from epoch 2, phase 1 is continue from where it stops at pre-epoch
                GA_i = GA_list[i]
                gBest_collection[i], gBest_fitness_collection[i] = GA_i.run()
                GA_list[i] = GA_i
                # print("gBest fitness of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        return gBest_collection, gBest_fitness_collection, GA_list

    def run_phase_2(self, gBest_collection, gBest=None):    # phase 2: running PSO on a set of gBests
                                                            # from each subswarm in phase 1
                                                            # the state of this phase will be ignored at the end of each
                                                            # epoch, only gBest is saved for next epoch
        PSO2 = PSO(self.fitness_function, self.dimension, self.m, gBest_collection, self.l2,
                   self.range0, self.range1, self.c3, self.c4)
        if gBest is not None:
            PSO2.set_gBest(gBest)
        gBest, fitness_gBest = PSO2.run()
        # print("##########")
        # print("gBest fitness of superswarm is {}".format(fitness_gBest))
        return gBest, fitness_gBest

    def run(self, subswarm_collection):

        GA_list = None
        gBest = None
        gBest_fitness_result = np.zeros(self.ep_max)
        start_time = time.clock()
        for i in range(self.ep_max):
            # print("start epoch {}................"
            #       ".............................."
            #       "..............................".format(i))
            gBest_collection, gBest_fitness_collection, GA_list = GSO.run_phase_1(subswarm_collection, GA_list)
            # print(gBest_fitness_collection)
            gBest, fitness_gBest_i = GSO.run_phase_2(gBest_collection, self.gBest)
            # print(fitness_gBest_i)
            if self.gBest is None:
                self.gBest = deepcopy(gBest)
            else:
                new_fitness = self.get_fitness(gBest)
                old_fitness = self.get_fitness(self.gBest)
                if new_fitness < old_fitness:
                    self.gBest = deepcopy(gBest)
            gBest_fitness_result[i] += fitness_gBest_i
            # print("end epoch {}................"
            #       ".............................."
            #       "..............................".format(i))

        total_time = time.clock() - start_time

        # print(gBest_fitness_result)
        # print(run_time_each_epoch)
        return gBest_fitness_result[-1], gBest_fitness_result, total_time


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

    m_s = parameter_set["GSO2"]['m']
    n_s = parameter_set["GSO2"]['n']
    l1_s = parameter_set["GSO2"]['l1']
    l2_s = parameter_set["GSO2"]['l2']
    max_ep_s = parameter_set["GSO2"]['max_ep']
    crossover_rates = parameter_set["GSO2"]["crossover_rate"]
    mutation_rates = parameter_set["GSO2"]["mutation_rate"]
    c3_s = parameter_set["GSO2"]["c3"]
    c4_s = parameter_set["GSO2"]["c4"]

    stability_number = 20
    combinations = []

    for fitness_function_name in fitness_function_names:
        if fitness_function_name in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f18', 'f19']:
            continue
        for dimension in dimensions:
            for range0 in range0_s:
                for range1 in range1_s:
                    if round(range0 + range1) != 0:
                        continue
                    for m in m_s:
                        for n in n_s:
                            for l1 in l1_s:
                                for l2 in l2_s:
                                    for crossover_rate in crossover_rates:
                                        for mutation_rate in mutation_rates:
                                            for c3 in c3_s:
                                                for c4 in c4_s:
                                                    for max_ep in max_ep_s:
                                                        function_evaluation = (m * n * l1 + m * l2) * max_ep
                                                        combination = [fitness_function_name, dimension, range0, range1,
                                                                       m, n, l1, l2, max_ep, crossover_rate,
                                                                       mutation_rate, c3, c4, function_evaluation]
                                                        if 30000 >= function_evaluation >= 25000:
                                                            combinations.append(combination)
    print(len(combinations))

    def save_result(combination, all_gbests, gBest_fitness, total_time):

        fitness_function_name = combination[0]
        path = '../results/' + str(fitness_function_name) + '/GSO2(GA+PSO)/'
        path1 = path + 'error_GSO2' + str(combination) + '.csv'
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
            columns = ['combination = [fitness_function_name, dimension, range0, range1, m, n, l1, l2, max_ep, '
                       'crossover_rate , mutation_rate, '
                       'c3, c4, function evaluation]',
                       'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, fitness_function_name, gBest_fitness, total_time):
        path = '../results/' + str(fitness_function_name) + '/GSO2(GA+PSO)/'
        path3 = path + 'stability_igso.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }

        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, m, n, l1, l2, max_ep, '
                       'crossover_rate , mutation_rate, '
                       'c3, c4, function evaluation]',
                       'total_time', 'gBest_fitness']
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

    model_name = "GSO2(GA+PSO)"

    for combination in combinations:
        fitness_function_name = combination[0]
        dimension = combination[1]
        range0 = combination[2]
        range1 = combination[3]
        m = combination[4]
        n = combination[5]
        l1 = combination[6]
        l2 = combination[7]
        max_ep = combination[8]
        crossover_rate = combination[9]
        mutation_rate = combination[10]
        c3 = combination[11]
        c4 = combination[12]

        path = "../results/" + fitness_function_name + "/" + model_name
        if not os.path.exists(path):
            os.mkdir(path)

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        GSO = GalacticSwarmOptimization(fitness_function, dimension, range0, range1, m, n, l1, l2, max_ep,
                                        crossover_rate, mutation_rate, c3, c4)
        subswarm_collection = GSO.init_population()
        fitness_gBest, gBest_fitness_collection, total_time = GSO.run(subswarm_collection)
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time: {}'.format(combination, fitness_gBest, total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            GSO_i = GalacticSwarmOptimization(fitness_function, dimension, range0, range1, m, n, l1, l2, max_ep,
                                              crossover_rate, mutation_rate, c3, c4)
            subswarm_collection = GSO_i.init_population()
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = GSO_i.run(subswarm_collection)
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_function_name, fitness_gBest, total_time)
