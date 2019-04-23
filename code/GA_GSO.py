import numpy as np
from copy import deepcopy

class GenericAlgorithm(object):

    def __init__(self, fitness_function, dimension, population_size, population, crossover_rate, mutation_rate, max_ep):
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.population_size = population_size
        self.population = population
        self.num_selected_parents = int(population_size/2)
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
        for i in range(self.max_ep):
            parents = self.select_mating_pool()
            self.population = self.choose_population(self.num_selected_parents)
            pair_list = self.choose_parent_pair(parents)

            for pair in pair_list:
                child1, child2 = self.crossover(pair)
                self.population = np.concatenate((self.population, self.mutation(child1),
                                                  self.mutation(child2)), axis=0)

        best_fitness = self.get_best_fitness()
        best_solution = self.get_index_chromosome_by_fitness(best_fitness, self.get_fitness(), self.population)

        return best_solution, best_fitness