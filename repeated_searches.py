import numpy
from prova import *
# questo file serve a generare quanti più problemi possibile e risolverli con varie euristiche per ottenere dati quali:
# costo, lunghezza soluzione, numero di nodi generati (esplorati), paths explored, tempo esecuzione

def data_same_matrix(grandezza: int, numero_matrici: int):
    with open("same_matrix%d.txt" %grandezza, "a") as file:
        for i in range(numero_matrici):
            # creo matrici e problema
            mat_in = Matrice(generate_matrix(grandezza))
            mat_fin = generate_matrix(grandezza)
            problema = Mproblem(mat_in, mat_fin)
            
            # creo dati soluzioni
            solution1 = execute3(str, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra_piu_costo_sol)
            solution2 = execute3(str, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra)
            solution3 = execute3(str, astar_search, problema, problema.posti_sbagliati)

            # scrivo nello stesso file
            file.write(f'Veloce {i}: tempo: {solution1["tempo"]}, nodi: {solution1["nodi"]}, explored_paths: {solution1["explored_paths"]}, cost: {solution1["cost"]}\n')
            file.write(f'Relaxed Pesata {i}: tempo: {solution2["tempo"]}, nodi: {solution2["nodi"]}, explored_paths: {solution2["explored_paths"]}, cost: {solution2["cost"]}\n')
            file.write(f'Relaxed {i}: tempo: {solution3["tempo"]}, nodi: {solution3["nodi"]}, explored_paths: {solution3["explored_paths"]}, cost: {solution3["cost"]}\n')

def data(grandezza: int, euristica:str):
    # scirvo in grandezza_euristica.txt, "a" sta per append
    with open("%d_%s.txt" %(grandezza, euristica), "a") as file:
        for i in range(36, 200):
            # creo matrici
            mat_in = Matrice(generate_matrix(grandezza))
            mat_fin = generate_matrix(grandezza)

            # stampo matrice
            print("matrice iniziale:\n")
            print(mat_in)
            print("matrice finale:\n")
            print(mat_fin)

            problema = Mproblem(mat_in, mat_fin)
            solution = dict()

            if euristica=="A-Star euristica veloce":
                solution = execute3(euristica, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra_piu_costo_sol)
            elif euristica=="A-Star euristica relaxed pesata":
                solution = execute3(euristica, astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra)
            elif euristica=="A-Star euristica relaxed":
                solution = execute3(str, astar_search, problema, problema.posti_sbagliati)
            else:
                print("errore")
                return
            file.write(f'{i}: tempo: {solution["tempo"]}, nodi: {solution["nodi"]}, explored_paths: {solution["explored_paths"]}, cost: {solution["cost"]}\n')
            print("risolti: %d" %i)

# decommenta se si vuole testare su matrici diverse
# for i in range(2, 7):
#     data(i, "A-Star euristica relaxed")
#     data(i, "A-Star euristica relaxed pesata")
#     data(i, "A-Star euristica veloce")

#data(6, "A-Star euristica veloce")
data(6, "A-Star euristica relaxed pesata")
data(6, "A-Star euristica relaxed")

# decommenta se si vuole testare su stessa matrice
# for i in range(2, 7):
#     data_same_matrix(i, 200)