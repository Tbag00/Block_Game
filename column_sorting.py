import numpy as np
from prova import generate_matrix
from prova import apply_gravity

res = generate_matrix()
print("matrice random\n", res)
# ordine crescente
# mat = np.sort(mat, axis=2)
# res = mat[:, :, 0]
res = res[::-1]  # ordine decrescente
print("matrice con colonna invertita\n", res)

apply_gravity(res)
print("matrice con gravita'\n", res)