import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import statistics

from aima.utils import manhattan_distance
from grafico import parse_file

data: dict

metric_names = ["indices", "tempo", "nodi", "explored_paths", "cost"]
colors = {"manhattan": "blue", "subgoal": "red", "relaxed": "green"}
euristic_names = ["manhattan", "subgoal", "relaxed"]
offsets = {"manhattan": -0.1, "subgoal": 0.0, "relaxed": 0.1}  # Per evitare sovrapposizioni

for metric in metric_names:
    for euristic in euristic_names:
        # liste contenenti media e varianza
        media = []
        std = []
        # per ogni dimensione di matrice
        for n in range(2,7):
            filename = f"same_matrix{n}.txt"  
            data = parse_file(filename)
            media.append(statistics.mean(data[euristic][metric]))
            std.append(statistics.stdev(data[euristic][metric]))
        
        x = [offsets[euristic] + i for i in range(2, 7)]  # distanzio leggermente i punti nelle x per renterli più visibili
        plt.title(f'Confronto {metric} tra euristiche')
        plt.xlabel('Dimensione matrice')
        plt.ylabel(metric.capitalize())
        plt.plot(x, media, label=metric, color=colors[euristic])
        plt.errorbar(x, media, yerr=std, capsize=3,  marker='o')

    plt.legend(euristic_names)
    plt.show()