import numpy as np
import time
import pandas as pd
import os.path
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
                    population[i, j] = np.random.uniform(range1-1, range1, 1)

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
            # self.best_solution, self.best_fitness = self.get_prey(population)
            new_best_solution, new_best_fitness = self.get_prey()
            if new_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(new_best_solution)
                self.best_fitness = deepcopy(new_best_fitness)
        return self.best_solution, self.get_fitness(self.best_solution)

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
        self.pBest = position
        self.gBest = np.random.uniform(range0, range1, varsize)
        self.temp = self.gBest

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(0, self.varsize)])

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
        self.gBest = None

    def init_population(self):  # initialize population by setting up randomly each subswarm
        subswarm_collection = []
        for i in range(self.m):
            subswarm_i = [np.random.uniform(self.range0, self.range1, self.dimension) for _ in range(self.n)]
            subswarm_collection.append(subswarm_i)
        return subswarm_collection

    def get_fitness(self, particle):
        return sum([particle[i]**2 for i in range(0, self.dimension)])

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
        WOA = WhaleOptimizationAlgorithm(self.dimension, self.m, gBest_collection, self.range0, self.range1, self.l2)
        if gBest is not None:
            WOA.set_best_solution(gBest)
        gBest, fitness_gBest = WOA.run()
        print("##########")
        print("gBest fitness of superswarm is {}".format(fitness_gBest))
        return gBest, fitness_gBest

    def run(self, subswarm_collection):

        PSO1_list = None
        gBest_fitness_result = np.zeros(self.ep_max)
        run_time_each_epoch = np.zeros(self.ep_max)

        for i in range(self.ep_max):
            start_time = time.clock()
            print("start epoch {}................"
                  ".............................."
                  "..............................".format(i))
            gBest_collection, gBest_fitness_collection, PSO1_list = GSO.run_phase_1(subswarm_collection, PSO1_list)
            gBest, fitness_gBest_i = GSO.run_phase_2(gBest_collection, self.gBest)
            if self.gBest is None:
                self.gBest = deepcopy(gBest)
            else:
                new_fitness = self.get_fitness(gBest)
                old_fitness = self.get_fitness(self.gBest)
                if new_fitness < old_fitness:
                    self.gBest = deepcopy(gBest)
            print(self.get_fitness(self.gBest))
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
    l1_list = [20, 50]
    l2_list = [150, 200, 400]
    ep_max = 10
    c1, c2, c3, c4 = 2.05, 2.05, 2.05, 2.05

    def save_result(combination, gBest_fitnes, avg_time_per_epoch):
        path = 'resultGSO(PSO-WOA).csv'
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



