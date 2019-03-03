import numpy as np
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