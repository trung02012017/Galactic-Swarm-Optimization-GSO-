import numpy as np
import json
import pandas as pd
import os
import operator
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sympy
import math

model_names = ["GSO1(PSO+PSO)", "GSO3(GSO+WOA)", "GSO4(GSO+OMWOA)", "IGSO(GSO+MWOA)", "MWOA"]
function_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16",
                  "f17","f21", "f22", "f23", "f24", "f25"]
# fig, ax = plt.subplots()

# for function_name in function_names:
#     for model_name in model_names:
#         print(model_name)
#         path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#         result_path = os.path.join(path, 'results', function_name, model_name)
#
#         model_result_path = os.path.join(result_path, os.listdir(result_path)[0])
#
#         model_result = pd.read_csv(model_result_path).values
#
#         model_log_result = np.zeros(model_result.shape)
#         print(model_log_result.shape)
#
#
#         for i in range(model_result.shape[0]):
#             if model_result[i, 1] != 0:
#                 b = np.log10(model_result[i, 1])
#             else:
#                 b = -330
#             model_log_result[i, 1] = b
#
#         if model_name in ["GSO1(PSO+PSO)", "GSO3(GSO+WOA)", "GSO4(GSO+OMWOA)", "IGSO(GSO+MWOA)"]:
#             model_log_result[300:305, 0] = np.arange(8000, 40001, 8000)
#             model_log_result[0:300, 0] = np.arange(2000, 8000, 20)
#         else:
#             model_log_result[:, 0] = np.arange(100, 40001, 100)
#
#         ax.plot(model_log_result[:, 0], model_log_result[:, 1], label=model_name)
#
#
#
#     ax.set_xlabel('Function Evaluation')
#     ax.set_ylabel('Global best fitness')
#     ax.grid()
#     plt.legend()
#
#     tick_points = [0, -50, -100, -150, -200, -250, -300, -330]
#     tick_names = ['10e0', '10e-50', '10e-100', '10e-150', '10e-200', '10e-250', '10e-300', '0']
#     plt.yticks(tick_points, tick_names)
#     name_fig = function_name + "_d_20_"
#     plt.title("function: " + function_name + " with dimension 20")
#     plt.savefig(name_fig, bbox_inches="tight")
#     plt.close('all')

def save_stability(dimension):

    for function_name in function_names:
        result = []
        for model_name in model_names:
            print(function_name, model_name, dimension)
            path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            result_path = os.path.join(path, 'results', function_name, model_name)

            model_stability_path = os.path.join(result_path, os.listdir(result_path)[-1])

            model_stability = pd.read_csv(model_stability_path).values

            if dimension==20:
                stability = model_stability[:20, 1]
            elif dimension==30:
                stability = model_stability[20:40, 1]
            elif dimension==50:
                stability = model_stability[40:60, 1]
            result.append([model_name, format(np.mean(stability), '.5e'), format(np.std(stability), '.5e')])

        df = pd.DataFrame(result, columns=["model", "mean", "std"])
        df.to_csv('../results/stability/d' + str(dimension) + '/' + function_name + '_dimension_' + str(dimension) +
                  '.csv', index=False)


if __name__ == '__main__':
    save_stability(20)
    save_stability(30)
    save_stability(50)
