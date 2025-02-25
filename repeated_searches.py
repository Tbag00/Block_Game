import numpy
from prova import *

# costo, lunghezza soluzione, numero di nodi generati (esplorati), paths explored, tempo esecuzione

mat1 = generate_matrix(6)
mat2 = generate_matrix(6)
print("matrice1:\n")
print(mat1)
print("matrice2:\n")
print(mat1)

mat_in = Matrice(mat1)

problema = Mproblem(mat_in, mat2)

dict = execute3("test", astar_search, problema, problema.posti_sbagliati_piu_giusti_sopra_piu_costo_sol)

print( dict)

