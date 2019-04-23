import numpy as np
import json
import os

class Fitness_Selector(object):

    def __init__(self):
        self.function_dict = None

    def f1(self, particle):
        return sum([particle[i]**2 for i in range(particle.shape[0])])

    def f2(self, particle):
        x = np.abs(particle)
        return np.sum(x) + np.prod(x)

    def f3(self, particle):
        fitness = 0
        for i in range(particle.shape[0]):
            for j in range(i+1):
              fitness += particle[j]
        return fitness

    def f4(self, particle):
        x = np.abs(particle)
        return np.max(x)

    def f5(self, particle):
        return sum([(100*((particle[i+1] - particle[i]**2)**2) + (particle[i] -1)**2) for i in range(particle.shape[0]-1)])

    def f6(self, particle):
        return np.sum([(particle[i] + 0.5)**2 for i in range(particle.shape[0])])

    def f7(self, particle):
        return np.sum([(i+1)*particle[i]**4 for i in range(particle.shape[0])]) + np.random.rand()

    def f8(self, particle):
        return particle[0]**2 + 10e6*np.sum([particle[i]**6 for i in range(1, particle.shape[0])])

    def f9(self, particle):
        return 10e6*particle[0]**2 + np.sum([particle[i]**6 for i in range(1, particle.shape[0])])

    def f10(self, particle):
        return (particle[0] - 1)**2 + np.sum([i*(2*particle[i]**2 - particle[i-1])**2 for i in range(1, particle.shape[0])])

    def f11(self, particle):
        return np.sum([((10e6)**((i)/(particle.shape[0]-1)))*particle[i]**2 for i in range(2, particle.shape[0])])

    def f12(self, particle):
        return np.sum([(i+1)*particle[i]**2 for i in range(particle.shape[0])])

    def f13(self, particle):
        return np.sum([particle[i]**2 for i in range(particle.shape[0])]) + \
               (np.sum([0.5*i*particle[i]**2 for i in range(particle.shape[0])]))**2 + \
               (np.sum([particle[i]**2 for i in range(particle.shape[0])]))**4

    def f14(self, particle):
        return np.sum([-1*particle[i]*np.sin(np.abs(particle[i])**0.5) for i in range(particle.shape[0])])

    def f15(self, particle):
        return np.sum([particle[i]**2 - 10*np.cos(2*np.pi*particle[i]) + 10 for i in range(particle.shape[0])])

    def f16(self, particle):
        return -20*np.exp(-0.2*(1/particle.shape[0]*np.sum([particle[i]**2 for i in range(particle.shape[0])]))**0.5) - \
               np.exp(1/particle.shape[0]*np.sum([np.cos(2*np.pi*particle[i]) for i in range(particle.shape[0])])) + 20 + \
               np.e
    def f17(self, particle):
        return 1/4000*np.sum([particle[i]**2 for i in range(particle.shape[0])]) - \
               np.prod([np.cos(particle[i]/((i+1)**0.5)) for i in range(particle.shape[0])]) + 1

    def f18(self, particle):
        return 0

    def f19(self, particle):
        return 0

    def f20(self, particle):
        return np.sum([((np.sum([(0.5**k)*np.cos(2*np.pi*(3**k)*(particle[i]+0.5)) for k in range(21)])) -
                       (particle.shape[0]*np.sum([(0.5**j)*np.cos(np.pi*(3**j)) for j in range(21)])))
                       for i in range(particle.shape[0])])

    def f21(self, particle):
        return np.sum([np.abs(particle[i]*np.sin(particle[i]) + 0.1*particle[i]) for i in range(particle.shape[0])])

    def f22(self, particle):
        return 0.5 + ((np.sin(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**2 - 0.5)*\
               (1+0.001*(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**-2

    def f23(self, particle):
        return 1/particle.shape[0]* \
               np.sum([particle[i]**4 - 16*particle[i]**2 + 5*particle[i] for i in range(particle.shape[0])])

    def f24(self, particle):
        return np.sum([particle[i]**2 + 2*particle[i+1]**2 - 0.3*np.cos(3*np.pi*particle[i]) -
                       0.4*np.cos(4*np.pi*particle[i+1]) + 0.7 for i in range(particle.shape[0]-1)])

    def f25(self, particle):
        return -1*(-0.1*np.sum([np.cos(5*np.pi*particle[i]) for i in range(particle.shape[0])]) -
                   np.sum([particle[i]**2 for i in range(particle.shape[0])]))

    def chose_function(self, function_name):
        function_dict = {
            'f1': self.f1,
            'f2': self.f2,
            'f3': self.f3,
            'f4': self.f4,
            'f5': self.f5,
            'f6': self.f6,
            'f7': self.f7,
            'f8': self.f8,
            'f9': self.f9,
            'f10': self.f10,
            'f11': self.f11,
            'f12': self.f12,
            'f13': self.f13,
            'f14': self.f14,
            'f15': self.f15,
            'f16': self.f16,
            'f17': self.f17,
            'f18': self.f18,
            'f19': self.f19,
            'f20': self.f20,
            'f21': self.f21,
            'f22': self.f22,
            'f23': self.f23,
            'f24': self.f24,
            'f25': self.f25,
        }
        return function_dict[function_name]


if __name__ == '__main__':

    # a = np.array([2, 3, 4])
    # fitness_selector = Fitness_Selector()
    # fitness_function = fitness_selector.chose_function('f1')
    # print(fitness_function(a))
    # print(round(1.1) - 1 )
    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup')
    print(params_path)
    for file in os.listdir(params_path):
        print(file)


#
# igso_parameter_set = data["parameters"]["IGSO"]
# m_s = igso_parameter_set['m']
# n_s = igso_parameter_set['n']
# l1_s = igso_parameter_set['l1']
# l2_s = igso_parameter_set['l2']
# max_ep_s = igso_parameter_set['max_ep']
#
# combinations = []
# for m in m_s:
#     for n in n_s:
#         for l1 in l1_s:
#             for l2 in l2_s:
#                 for max_ep in max_ep_s:
#                     combination = [m, n, l1, l2, max_ep]
#                     function_evaluation = (m*n*l1 + m*l2)*max_ep
#                     if function_evaluation >= 50000 and function_evaluation <= 60000:
#                         print("combination: {} and function evaluation: {}".format(str(combination),
#                                                                                    function_evaluation))
#                         combinations.append(combination)
# print(len(combinations))