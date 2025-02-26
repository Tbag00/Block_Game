import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import statistics

from aima.utils import manhattan_distance
from grafico import parse_file

data: dict
# y = data["manhattan"]["tempo"]
# plt.plot(range(100), y)
# plt.show()

metric_names = ["indices", "tempo", "nodi", "explored_paths", "cost"]
colors = {"relaxed": "green", "subgoal": "red", "manhattan": "blue"}
euristic_names = {"manhattan", "subgoal", "relaxed"}

for metric in metric_names:
    for euristic in euristic_names:
        # liste contenenti media e varianza
        media = []
        std = []
        for n in range(2,7):
            filename = f"same_matrix{n}.txt"  
            data = parse_file(filename)
            media.append(statistics.mean(data[euristic][metric]))
            std.append(statistics.stdev(data[euristic][metric]))
                
        plt.title(f'Confronto {metric} tra categorie')
        plt.xlabel('Dimensione matrice')
        plt.ylabel(metric.capitalize())
        plt.plot(range(2, 7), media, label=metric, color=colors[euristic])
        plt.errorbar(range(2, 7), media, yerr=std, capsize=5,  marker='o')
    plt.legend(euristic_names)
    plt.show()