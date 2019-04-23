import numpy as np
import time
import pandas as pd
import os.path
import math
from copy import deepcopy
import json
from fitness_selector import Fitness_Selector
from PSO_GSO import PSO
from OMWOA_GSO import ModifiedWOA

class GalacticSwarmOptimization(object):

    def __init__(self, fitness_function, dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2):
        self.fitness_function = fitness_function    # fitness function predefined
        self.dimension = dimension      # dimension size
        self.range0 = range0            # lower boundary of the value for each dimension
        self.range1 = range1            # upper boundary of the value for each dimension
        self.m = m                      # the number of subswarms that the population is divided into
        self.n = n                      # the number of particles per each subswarm ( so population size = m x n )
        self.l1 = l1                    # the number of epochs of PSO in phase 1
        self.l2 = l2                    # the number of epochs of PSO in phase 2
        self.ep_max = ep_max            # the number of epochs of PSO the whole system
        self.c1 = c1                    # c1, c2 is parameters for the formula in phase 1
        self.c2 = c2                    # c3, c4 is parameters for the formula in phase 2
        self.gBest = None

    def init_population(self):  # initialize population by setting up randomly each subswarm
        subswarm_collection = []
        for i in range(self.m):
            subswarm_i = [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.n)]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def run_phase_1(self, subswarm_collection, PSO1_list=None): # run PSO in subswarms independently
        gBest_collection = np.zeros((self.m, self.dimension))   # set of gBests of all subswarms after running PSO
        gBest_fitness_collection = np.zeros(self.m)             # set of all gBest fitness (just for showing the result)
        if PSO1_list is None:   # at epoch 1, PSO objects are created, at the end of each epoch,
                                # the states of each subswarm is saved and continued in next epoch
            PSO1_list = []
            for i in range(self.m):
                subswarm_i = subswarm_collection[i]
                PSO1_i = PSO(self.fitness_function, self.dimension, self.n, subswarm_i, self.l1, self.range0,
                             self.range1, self.c1, self.c2)
                gBest_collection[i], gBest_fitness_collection[i], _ = PSO1_i.run()
                PSO1_list.append(PSO1_i)
                # print("gBest of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        else:
            for i in range(self.m): # from epoch 2, phase 1 is continue from where it stops at pre-epoch
                PSO1_i = PSO1_list[i]
                gBest_collection[i], gBest_fitness_collection[i], _ = PSO1_i.run()
                PSO1_list[i] = PSO1_i
                # print("gBest fitness of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        return gBest_collection, gBest_fitness_collection, PSO1_list

    def run_phase_2(self, gBest_collection, gBest=None):    # phase 2: running PSO on a set of gBests
                                                            # from each subswarm in phase 1
                                                            # the state of this phase will be ignored at the end of each
                                                            # epoch, only gBest is saved for next epoch
        WOA = ModifiedWOA(self.fitness_function, self.dimension, self.m, gBest_collection, self.range0, self.range1,
                          self.l2)
        if gBest is not None:
            WOA.set_best_solution(gBest)
        gBest, fitness_gBest, results = WOA.run()
        return gBest, fitness_gBest, results

    def run(self, subswarm_collection):

        PSO1_list = None
        gBest_fitness_result = np.zeros(self.l2 + self.ep_max)
        start_time = time.clock()

        for i in range(self.ep_max):
            # print("start epoch {}................"
            #       ".............................."
            #       "..............................".format(i))
            gBest_collection, gBest_fitness_collection, PSO1_list = GSO.run_phase_1(subswarm_collection, PSO1_list)
            gBest, fitness_gBest_i, results = GSO.run_phase_2(gBest_collection, self.gBest)
            if self.gBest is None:
                self.gBest = deepcopy(gBest)
            else:
                new_fitness = self.get_fitness(gBest)
                old_fitness = self.get_fitness(self.gBest)
                if new_fitness < old_fitness:
                    self.gBest = deepcopy(gBest)
            # print("gBest of superswarm is {}".format(self.get_fitness(self.gBest)))
            if i == 0:
                gBest_fitness_result[0:self.l2] += results
            gBest_fitness_result[self.l2+i] = self.get_fitness(self.gBest)
            # print("end epoch {}................"
            #       ".............................."
            #       "..............................".format(i))
        total_time = time.clock() - start_time
        # print(self.gBest)

        return gBest_fitness_result[-1], gBest_fitness_result, total_time


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup', 'params_demo.json')

    with open(params_path, 'r') as f:
        data = json.load(f)

    igso_parameter_set = data["parameters"]

    fitness_function_names = igso_parameter_set['function']
    dimensions = igso_parameter_set["dimension"]
    range0_s = igso_parameter_set["range0"]
    range1_s = igso_parameter_set["range1"]

    m_s = igso_parameter_set["IGSO"]['m']
    n_s = igso_parameter_set["IGSO"]['n']
    l1_s = igso_parameter_set["IGSO"]['l1']
    l2_s = igso_parameter_set["IGSO"]['l2']
    max_ep_s = igso_parameter_set["IGSO"]['max_ep']
    c1_s = igso_parameter_set["IGSO"]["c1"]
    c2_s = igso_parameter_set["IGSO"]["c2"]

    stability_number = 20
    combinations = []

    for fitness_function_name in fitness_function_names:
        if fitness_function_name in ['f18', 'f19']:
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
            for m in m_s:
                for n in n_s:
                    for l1 in l1_s:
                        for l2 in l2_s:
                            for c1 in c1_s:
                                for c2 in c2_s:
                                    for max_ep in max_ep_s:
                                        function_evaluation = (m * n * l1 + m * l2) * max_ep
                                        combination = [fitness_function_name, dimension, range0, range1,
                                                       m, n, l1, l2, max_ep, c1, c2, function_evaluation]
                                        combinations.append(combination)
    print(len(combinations))

    def save_result(combination, all_gbests, gBest_fitness, total_time):

        fitness_function_name = combination[0]
        path = '../results/' + str(fitness_function_name) + '/GSO4(GSO+OMWOA)/'
        path1 = path + 'error_GSO4' + str(combination) + '.csv'
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
            columns = ['combination = [fitness_function_name, dimension, range0, range1, m, n, l1, l2, max_ep, c1 ,c2, '
                       'function evaluation]',
                       'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, fitness_function_name, gBest_fitness, total_time):
        path = '../results/' + str(fitness_function_name) + '/GSO4(GSO+OMWOA)/'
        path3 = path + 'stability_gso4.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }

        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, m, n, l1, l2, max_ep, c1, c2, '
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

    model_name = "GSO4(GSO+OMWOA)"

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
        c1 = combination[9]
        c2 = combination[10]

        path = "../results/" + fitness_function_name
        if not os.path.exists(path):
            os.mkdir(path)

        path = "../results/" + fitness_function_name + "/" + model_name
        if not os.path.exists(path):
            os.mkdir(path)

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        GSO = GalacticSwarmOptimization(fitness_function, dimension, range0, range1, m, n, l1, l2, max_ep, c1, c2)
        subswarm_collection = GSO.init_population()
        fitness_gBest, gBest_fitness_collection, total_time = GSO.run(subswarm_collection)
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time: {}'.format(combination, fitness_gBest, total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            GSO_i = GalacticSwarmOptimization(fitness_function, dimension, range0, range1, m, n, l1, l2, max_ep, c1, c2)
            subswarm_collection = GSO_i.init_population()
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = GSO_i.run(subswarm_collection)
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_function_name, fitness_gBest, total_time)



