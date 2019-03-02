import numpy as np
from operator import itemgetter
from copy import deepcopy
from math import gamma
from spiral_forms import logarithm_spiral_curve,archimedes_spiral_curve

class LevyWOA:
    """
    Apply Levy flight in whale optimization algorithm
    """
    def __init__(self, num_agent, problem_size, search_space):

        self.num_agent = num_agent
        self.problem_size = problem_size
        self.search_space = search_space
    
    def init_pop(self):
        """ 
        Initialize population 
        """
        pop = []
        for _ in range(self.num_agent):
            solution = np.random.uniform(self.search_space[0][0], self.search_space[0][1], self.problem_size)
            fit = self.fitness(solution)
            pop.append([solution,fit])
        return pop
    
    def fitness(self, solution):
        """
        Calculate fitness of a solution
        
        """
        fit = 0
        for i in range(self.problem_size):
            if i%2 == 1 :   
                fit += np.square(solution[i])
            else:
                fit += np.power(solution[i],2)
        return fit        
    
    def identify_worst_index(self, pop):
        """
        find the index of the worst solution 
        """
        worst_fit = pop[0][1]
        worst_index = 0
        for i in range(self.num_agent):
            if worst_fit < pop[i][1]:
                worst_fit = pop[i][1]
                worst_index = i
        return worst_index

    def opposition_based_replace(self):
        """
        Replace the worst solution by a random solution 
        Random solution and it's opposite solution are chosen and compared.
        The better one will be chosen as the replacement for the worst solution in the population 
        
        Return better solution between two candidate 
        """
        candidate_1 = self.create_random_whale()
        #the opposite solution of candidate 1
        candidate_2 = [ self.search_space[i][0] + self.search_space[i][1] - candidate_1[i] for i in range(self.problem_size)]
        #make candidate 2 become numpy array 
        candidate_2 = np.array(candidate_2)
        
        if self.fitness(candidate_1) < self.fitness(candidate_2):
            return candidate_1
        else:
            return  candidate_2

    def create_random_whale(self):
        """
        Create a random solution 
        """
        return np.random.uniform(self.search_space[0][0], self.search_space[0][1], self.problem_size)

    def identify_prey_index(self, pop):
        """
        Find index of the best solution 
        """
        prey_fit = pop[0][1]
        prey_index = 0
        for i in range(self.num_agent):
            if prey_fit > pop[i][1] :
                prey_fit = pop[i][1]
                prey_index = i
        return prey_index

    def amend(self, solution1):
        """
        Check and repair duplicate genes to make sure each search agent is valid
        return the valid solution 
        """
        solution = solution1
        for i in range(self.problem_size):
            if solution[i] <  self.search_space[i][0]:
                solution[i] = self.search_space[i][0]
               
            if solution[i] > self.search_space[i][1]:
                solution[i] = self.search_space[i][1]
        return solution
    
    def search(self, max_iter):
        """
        Search for the optimum solution

        """
        pop = self.init_pop()
        #find prey
        prey = sorted(pop, key=lambda item: item[1])[0]
        #Main loop
        for i in range(max_iter):
            a = 2*np.cos(i/max_iter)
            beta = 1
            #muy and v are two random variables which follow normal distribution
            #sigma_muy : standard deviation of muy 
            sigma_muy = np.power(gamma(1+beta)*np.sin(np.pi*beta/2)/(gamma((1+beta)/2)*beta*np.power(2,(beta-1)/2)), 1/beta)
            #sigma_v : standard deviation of v
            sigma_v = 1
            for j in range(self.num_agent):
                r = np.random.rand()
                A = 2*a*r - a
                C = 2*r
                p = np.random.rand() #random number to select encircling prey or spiral updating position
                b = 1    
                if (p < 0.5) : # if p < 0.5 , encircling prey is applied
                    if np.abs(A) < 1: # if A< 1: exploitation phase
                        # modified woa: levy flight, need read paper to undertand parameter 
                        muy = np.random.normal(0,sigma_muy)
                        v = np.random.normal(0, sigma_v)
                        s = muy/np.power(np.abs(v),1/beta)    
                        D = self.create_random_whale()
                        LB = 0.01*s*(pop[j][0] - prey[0])
                        pop[j][0] = D*LB #update new position 
                        pop[j][0] = self.amend(pop[j][0]) #check and repair  
                        pop[j][1] = self.fitness(pop[j][0]) # recalculate fitness 
                    else : # if  A>1 : exploration phase
                        # select a random whale to be prey 
                        x_rand = pop[np.random.randint(self.num_agent)] 
                        D = np.abs(C*x_rand[0] - pop[j][0])
                        pop[j][0] = self.amend(x_rand[0] - A*D)
                        pop[j][1] = self.fitness(pop[j][0])
                else:   # if p>0.5 spiral updating position is applied
                        D1 = np.abs(prey[0] -pop[j][0])
                        pop[j][0] = logarithm_spiral_curve(D1,prey[0],b)
                        pop[j][0] = self.amend(pop[j][0])
                        pop[j][1] = self.fitness(pop[j][0])
            # update prey
            current_best = sorted(pop, key=lambda item: item[1])[0]
            if current_best[1] < prey[1]:
                prey = deepcopy(current_best)
            print("best solution until iter {}  has  fitness {} ".format(i, prey[1]))
        print("best solution is {} \n with fitness {} ".format(prey[0],prey[1]))
                   
if __name__ == "__main__":

    problem_size = 100
    num_agent = int(problem_size)
    search_space = []
    for i in range(problem_size):
        search_space.append([-1,1])
    max_iter = 100
    woa = LevyWOA(num_agent, problem_size, search_space)
    woa.search(max_iter)

        