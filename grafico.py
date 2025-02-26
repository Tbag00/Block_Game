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

def plot_data(data):
    metrics = ["tempo", "nodi", "explored_paths", "cost"]
    colors = {"manhattan": "blue", "subgoal": "red", "relaxed": "green"}
    offsets = {"manhattan": -0.1, "subgoal": 0.0, "relaxed": 0.1}  # Per evitare sovrapposizioni
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for category, values in data.items():
            indices = np.array(values["indices"]) #+ offsets[category]  # Aggiunta di offset per differenziare linee sovrapposte
            plt.plot(indices, values[metric],'.', label=category, color=colors[category])
            
        
        plt.title(f'Confronto {metric} tra categorie')
        plt.xlabel('Esecuzione')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    filename = "same_matrix6.txt"  # Modifica con il percorso corretto se necessario
    data = parse_file(filename)
    plot_data(data)
