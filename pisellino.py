import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

def parse_file(filename, chunk_size=1000):
    data = {"manhattan": defaultdict(list), "subgoal": defaultdict(list), "relaxed": defaultdict(list)}
    
    with open(filename, 'r') as file:
        chunk = file.readlines(chunk_size)  # Leggiamo il file a blocchi
        while chunk:
            for line in chunk:
                match = re.match(r'([^\d]+)(\d+): tempo: ([\d\.]+), nodi: (\d+), explored_paths: (\d+), cost: (\d+)', line)
                if match:
                    category, index, tempo, nodi, explored_paths, cost = match.groups()
                    category = category.strip()
                    if category in data:
                        data[category]["indices"].append(int(index))
                        data[category]["tempo"].append(float(tempo))
                        data[category]["nodi"].append(int(nodi))
                        data[category]["explored_paths"].append(int(explored_paths))
                        data[category]["cost"].append(int(cost))
            chunk = file.readlines(chunk_size)  # Continua a leggere i blocchi successivi
    
    return data

def plot_scatter(data):
    metrics = ["tempo", "nodi", "explored_paths", "cost"]
    colors = {"manhattan": "blue", "subgoal": "red", "relaxed": "green"}
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for category, values in data.items():
            indices = np.array(values["indices"])
            plt.scatter(indices, values[metric], label=category, color=colors[category], alpha=0.7, s=20)

        plt.xlim(min(indices) - 5, max(indices) + 5)
        plt.ylim(0, max(max(values[metric]) for values in data.values()) * 1.1)

        plt.grid(True, linestyle='--', alpha=0.6)

        plt.xlabel("Esecuzione", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Confronto {metric} tra categorie', fontsize=14)
        plt.legend()
        plt.show()

def plot_histograms(data, bins=10):
    metrics = ["tempo", "nodi", "explored_paths", "cost"]
    colors = {"manhattan": "blue", "subgoal": "red", "relaxed": "green"}
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        contrasto = 0.15
        for category, values in data.items():
            plt.hist(values[metric], bins=bins, alpha=contrasto, label=category, color=colors[category], edgecolor="black")
            contrasto +=0.1
        plt.xlabel(metric.capitalize(), fontsize=12)
        plt.ylabel("Frequenza", fontsize=12)
        plt.title(f'Istoogramma di {metric} per categorie', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

if __name__ == "__main__":
    filename = "same_matrix6.txt"  
    data = parse_file(filename)

    # Scatter plot
    plot_scatter(data)

    # Istogrammi sovrapposti
    plot_histograms(data, bins="scott")
