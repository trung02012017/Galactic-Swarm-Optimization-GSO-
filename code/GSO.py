import numpy as np
import time
import pandas as pd
import os.path
from copy import deepcopy


class PSO(object):

    def __init__(self, varsize, swarmsize, position, epochs, range0, range1, c1, c2):
        self.varsize = varsize
        self.swarmsize = swarmsize
        self.epochs = epochs
        self.range0 = range0
        self.range1 = range1
        self.c1 = c1
        self.c2 = c2
        self.position = position
        self.velocity = np.zeros((swarmsize, varsize))
        self.pBest = [np.random.uniform(range0, range1, varsize) for _ in range(swarmsize)]
        self.gBest = np.random.uniform(range0, range1, varsize)
        self.temp = self.gBest

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(0, self.varsize)])
        # fitness = 0
        # for j in range(self.varsize):
        #     if j%2 == 0:
        #        fitness += particle[j]**3
        #     else:
        #         fitness += particle[j]**2
        # return fitness


    def set_gBest(self, gBest):
        self.gBest = gBest

    def run(self):

        v_max = 10
        w_max = 0.9
        w_min = 0.4
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
        return self.gBest, self.get_fitness(self.gBest)


class GalacticSwarmOptimization(object):

    def __init__(self, dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2, c3, c4):
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
        self.c3 = c3
        self.c4 = c4

    def init_population(self):  # initialize population by setting up randomly each subswarm
        subswarm_collection = []
        for i in range(self.m):
            subswarm_i = [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.n)]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection


    def run_phase_1(self, subswarm_collection, PSO1_list=None): # run PSO in subswarms independently
        gBest_collection = np.zeros((self.m, self.dimension))   # set of gBests of all subswarms after running PSO
        gBest_fitness_collection = np.zeros(self.m)             # set of all gBest fitness (just for showing the result)
        if PSO1_list is None:   # at epoch 1, PSO objects are created, at the end of each epoch,
                                # the states of each subswarm is saved and continued in next epoch
            PSO1_list = []
            for i in range(self.m):
                subswarm_i = subswarm_collection[i]
                PSO1_i = PSO(self.dimension, self.n, subswarm_i, self.l1, self.range0, self.range1, self.c1, self.c2)
                gBest_collection[i], gBest_fitness_collection[i] = PSO1_i.run()
                PSO1_list.append(PSO1_i)
                print("gBest of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        else:
            for i in range(self.m): # from epoch 2, phase 1 is continue from where it stops at pre-epoch
                PSO1_i = PSO1_list[i]
                gBest_collection[i], gBest_fitness_collection[i] = PSO1_i.run()
                PSO1_list[i] = PSO1_i
                print("gBest fitness of subswarm {} is {}".format(i, gBest_fitness_collection[i]))
        return gBest_collection, gBest_fitness_collection, PSO1_list

    def run_phase_2(self, gBest_collection, gBest=None):    # phase 2: running PSO on a set of gBests
                                                            # from each subswarm in phase 1
                                                            # the state of this phase will be ignored at the end of each
                                                            # epoch, only gBest is saved for next epoch
        PSO2 = PSO(self.dimension, self.m, gBest_collection, self.l2, self.range0, self.range1, self.c3, self.c4)
        if gBest is not None:
            PSO2.set_gBest(gBest)
        gBest, fitness_gBest = PSO2.run()
        print("##########")
        print("gBest fitness of superswarm is {}".format(fitness_gBest))
        return gBest, fitness_gBest

    def run(self, subswarm_collection):

        PSO1_list = None
        gBest = None
        gBest_fitness_result = np.zeros(self.ep_max)
        run_time_each_epoch = np.zeros(self.ep_max)

        for i in range(self.ep_max):
            start_time = time.clock()
            print("start epoch {}................"
                  ".............................."
                  "..............................".format(i))
            gBest_collection, gBest_fitness_collection, PSO1_list = GSO.run_phase_1(subswarm_collection, PSO1_list)
            gBest, fitness_gBest_i = GSO.run_phase_2(gBest_collection, gBest)
            gBest_fitness_result[i] += fitness_gBest_i
            print("end epoch {}................"
                  ".............................."
                  "..............................".format(i))
            run_time = time.clock() - start_time
            run_time_each_epoch[i] += run_time
            avg_time_per_epoch = np.mean(run_time_each_epoch)

        print(gBest_fitness_result)
        print(run_time_each_epoch)
        return gBest_fitness_result[-1], avg_time_per_epoch


if __name__ == '__main__':

    dimension = 50
    range0 = -10
    range1 = 10
    m_list = [15, 20, 25]
    n_list = [5, 10]
    l1_list = [50, 100]
    l2_list = [500, 1000]
    ep_max = 20
    c1, c2, c3, c4 = 2.05, 2.05, 2.05, 2.05

    def save_result(combination, gBest_fitnes, avg_time_per_epoch):
        path = 'result1.csv'
        combination = [combination]
        result = {
            'combination': combination,
            'gBest_fitness': format(gBest_fitnes, '.2e'),
            'avg_time_per_epoch': round(avg_time_per_epoch, 2)
        }

        df = pd.DataFrame(result)
        if not os.path.exists(path):
            columns = ['combination [m, n, l1, l2, ep_max, c1, c2, c3, c4]', 'gBest_fitness', 'avg_time_per_epoch']
            df.columns = columns
            df.to_csv(path, index=False, columns=columns)
        else:
            with open(path, 'a') as csv_file:
                df.to_csv(csv_file, mode='a', header=False, index=False)




    combinations = []
    for m in m_list:
        for n in n_list:
            for l1 in l1_list:
                for l2 in l2_list:
                    combination = [m, n, l1, l2, ep_max, c1, c2, c3, c4]
                    combinations.append(combination)

    for combination in combinations:
        m = combination[0]
        n = combination[1]
        l1 = combination[2]
        l2 = combination[3]

        GSO = GalacticSwarmOptimization(dimension, range0, range1, m, n, l1, l2, ep_max, c1, c2, c3, c4)
        subswarm_collection = GSO.init_population()
        gBest_fitness, avg_time_per_epoch = GSO.run(subswarm_collection)
        save_result(combination, gBest_fitness, avg_time_per_epoch)



