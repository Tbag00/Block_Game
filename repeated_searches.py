import numpy
from prova import *
# questo file serve a generare quanti più problemi possibile e risolverli con varie euristiche per ottenere dati quali:
# costo, lunghezza soluzione, numero di nodi generati (esplorati), paths explored, tempo esecuzione

def data(grandezza: int, euristica:str):
    # scirvo in grandezza_euristica.txt, "a" sta per append
    with open("%d_%s.txt" %(grandezza, euristica), "a") as file:
        for i in range(200):
            # creo matrici
            mat_in = Matrice(generate_matrix(grandezza))
            mat_fin = generate_matrix(grandezza)

            # stampo matrice
            # print("matrice iniziale:\n")
            # print(mat_in)
            # print("matrice finale:\n")
            # print(mat_fin)

            problema = Mproblem(mat_in, mat_fin)
            solution = dict()

            if euristica=="A-Star euristica veloce":
                solution = execute3(str, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra_piu_costo_sol)
            elif euristica=="A-Star euristica relaxed pesata":
                solution = execute3(str, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra)
            elif euristica=="A-Star euristica relaxed":
                solution = execute3(str, astar_search, problema, problema.posti_sbagliati)
            else:
                print("errore")
                return
            file.write(f'{i}: tempo: {solution["tempo"]}, nodi: {solution["nodi"]}, explored_paths: {solution["explored_paths"]}, cost: {solution["cost"]}\n')

for i in range(2, 6):
    data(i, "A-Star euristica relaxed")
    data(i, "A-Star euristica relaxed pesata")
    data(i, "A-Star euristica veloce")