import numpy as np
import time
import pandas as pd
import os.path
import json
from fitness_selector import Fitness_Selector

class GenericAlgorithm(object):

    def __init__(self, fitness_function, dimension, population_size, population, num_selected_parents, crossover_rate,
                 mutation_rate, max_ep):
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.population_size = population_size
        self.population = population
        self.num_selected_parents = num_selected_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_ep = max_ep

    def choose_population(self, num_citizens):
        num_delete = self.population_size - num_citizens
        random = np.random.permutation(self.population_size)
        delete_index = random[0:num_delete]
        population = np.delete(self.population, delete_index, 0)

        return population

    def get_fitness(self):
        fitness_value = np.zeros((self.population_size, 1))
        for i in range(self.population_size):
            fitness = self.fitness_function(self.population[i])
            fitness_value[i] = fitness
        return fitness_value

    def get_best_fitness(self):

        fitness_value = self.get_fitness()
        indices = np.where(fitness_value == np.min(fitness_value))[0]
        index = indices[0]
        # print(indices)
        best_solution = self.population[index]
        best_fitness = fitness_value[index]
        # print(best_fitness)
        return best_fitness

    def get_index_chromosome_by_fitness(self, value, fitness_value, population):
        index = np.where(fitness_value == value)[0][0]
        # print(index)
        chromosome = population[index]
        return chromosome

    def select_mating_pool(self):

        selected_parents = np.zeros((self.num_selected_parents, self.dimension))

        fitness_value = self.get_fitness()
        # print(fitness_value)
        sorted_fitness = np.sort(fitness_value, axis=0)
        # print(sorted_fitness)
        for i in range(self.num_selected_parents):
            parent = self.get_index_chromosome_by_fitness(sorted_fitness[i], fitness_value, self.population)
            selected_parents[i] += parent

        return selected_parents

    def choose_parent_pair(self, parents):
        size = parents.shape[0]
        parent_pair_list = []
        parents_indices = np.random.permutation(size)
        for i in range(0, size, 2):
            parent1_index = parents_indices[i]
            parent2_index = parents_indices[i-1]
            parent1 = parents[parent1_index].reshape((1, self.dimension))
            parent2 = parents[parent2_index].reshape((1, self.dimension))
            pair = [parent1, parent2]
            parent_pair_list.append(pair)

        return parent_pair_list

    def crossover(self, parent_pair):

        parent1 = parent_pair[0]
        parent2 = parent_pair[1]

        child1 = np.zeros((1, self.dimension))
        child2 = np.zeros((1, self.dimension))

        num_gens_parent1 = int(self.dimension*self.crossover_rate)
        num_gens_parent2 = self.dimension - num_gens_parent1

        permutation = np.random.permutation(self.dimension)

        for i in range(num_gens_parent1):
            index = permutation[i]
            child1[:, index] += parent1[:, index]
            child2[:, index] += parent2[:, index]

        permutation = permutation[num_gens_parent1:self.dimension]

        for i in permutation:
            index = i
            child1[:, index] += parent2[:, index]
            child2[:, index] += parent1[:, index]

        return child1, child2

    def mutation(self, child):

        a = np.random.randint(0, self.dimension-1, 2)
        a1 = a[0]
        # print(a1)
        range = int(self.mutation_rate*self.population_size)
        if a1+range > self.population_size:
            a2 = int(a1 + range)
        else:
            a2 = int(a1 - range)

        if a1 < a2:
            selected_part = child[:, a1:a2]
            reversed_part = np.flip(selected_part)
            child[:, a1:a2] = reversed_part

        if a1 > a2:
            selected_part = child[:, a2:a1]
            reversed_part = np.flip(selected_part)
            child[:, a2:a1] = reversed_part

        return child

    def run(self):
        best_fitness_collection = np.zeros(self.max_ep)
        start_time = time.clock()
        for i in range(self.max_ep):
            parents = self.select_mating_pool()
            self.population = self.choose_population(self.num_selected_parents)
            pair_list = self.choose_parent_pair(parents)

            for pair in pair_list:
                child1, child2 = self.crossover(pair)
                self.population = np.concatenate((self.population, self.mutation(child1),
                                                  self.mutation(child2)), axis=0)

            best_fitness = np.round(self.get_best_fitness(), 3)
            # print(self.get_index_chromosome_by_fitness(best_fitness, self.get_fitness(), self.population))
            best_fitness_collection[i] = best_fitness
        total_time = time.clock() - start_time

        return best_fitness_collection[-1], best_fitness_collection, total_time


if __name__ == "__main__":

    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup', 'params.json')

    with open(params_path, 'r') as f:
        data = json.load(f)

    parameter_set = data["parameters"]

    fitness_function_names = parameter_set['function']
    dimensions = parameter_set["dimension"]
    range0_s = parameter_set["range0"]
    range1_s = parameter_set["range1"]

    population_sizes = parameter_set["GA"]['population_size']
    crossover_rates = parameter_set["GA"]['crossover_rate']
    mutation_rates = parameter_set["GA"]['mutation_rate']
    max_eps = parameter_set["GA"]['max_ep']

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
                        for crossover_rate in crossover_rates:
                            for mutation_rate in mutation_rates:
                                for max_ep in max_eps:
                                    function_evaluation = population_size * max_ep
                                    combination = [fitness_function_name, dimension, range0, range1,
                                                   population_size, crossover_rate, mutation_rate, max_ep,
                                                   function_evaluation]
                                    if 30000 >= function_evaluation >= 25000:
                                        combinations.append(combination)
    print(len(combinations))


    def save_result(combination, all_gbests, gBest_fitness, total_time):
        fitness_function_name = combination[0]
        path = '../results/' + str(fitness_function_name) + '/GA/'
        path1 = path + 'error_GAA' + str(combination) + '.csv'
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
                       'crossover_rate, mutation_rate, function evaluation]',
                       'total_time', 'gBest_fitness']
            df_models_log.columns = columns
            df_models_log.to_csv(path2, index=False, columns=columns)
        else:
            with open(path2, 'a') as csv_file:
                df_models_log.to_csv(csv_file, mode='a', header=False, index=False)


    def save_result_stability(params, fitness_function_name, gBest_fitness, total_time):
        path = '../results/' + str(fitness_function_name) + '/GA/'
        path3 = path + 'stability_igso.csv'
        stability = {
            'combination': params,
            'gBest_fitness': gBest_fitness,
            'total_time': total_time,
        }

        df_stability = pd.DataFrame(stability)
        if not os.path.exists(path3):
            columns = ['combination = [fitness_function_name, dimension, range0, range1, population_size, max_ep, '
                       'crossover_rate, mutation_rate, function evaluation]',
                       'gBest_fitness', 'total_time']
            df_stability.columns = columns
            df_stability.to_csv(path3, index=False, columns=columns)
        else:
            with open(path3, 'a') as csv_file:
                df_stability.to_csv(csv_file, mode='a', header=False, index=False)

    model_name = "GA"

    for combination in combinations:
        fitness_function_name = combination[0]
        dimension = combination[1]
        range0 = combination[2]
        range1 = combination[3]
        population_size = combination[4]
        num_selected_parents = int(population_size/2)
        crossover_rate = combination[5]
        mutation_rate = combination[6]
        max_ep = combination[7]

        path = "../results/" + fitness_function_name + "/" + model_name
        if not os.path.exists(path):
            os.mkdir(path)

        fitness_selector = Fitness_Selector()
        fitness_function = fitness_selector.chose_function(fitness_function_name)

        population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
        GA = GenericAlgorithm(fitness_function, dimension, population_size, population, num_selected_parents,
                              crossover_rate, mutation_rate, max_ep)
        fitness_gBest, gBest_fitness_collection, total_time = GA.run()
        save_result(combination, gBest_fitness_collection, fitness_gBest, total_time)
        print('combination:{} and gBest fitness: {} and total time: {}'.format(combination, fitness_gBest, total_time))

        params = []
        fitness_gBest = np.zeros(stability_number)
        total_time = np.zeros(stability_number)
        for i in range(stability_number):
            population_i = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]
            GA_i = GenericAlgorithm(fitness_function, dimension, population_size, population_i, num_selected_parents,
                                  crossover_rate, mutation_rate, max_ep)
            fitness_gBest_i, gBest_fitness_collection_i, total_time_i = GA_i.run()
            fitness_gBest[i] += fitness_gBest_i
            total_time[i] += total_time_i
            params.append(str(combination))
        save_result_stability(params, fitness_function_name, fitness_gBest, total_time)
