import numpy as np
mat_i = np.array([[1,0,0],[2,0,0],[3,0,0]])
mat_f = np.array([[0,0,0],[0,2,0],[0,3,1]])
num = np.sum((mat_i != 0) & (mat_i != mat_f))
goal_f = {mat_f[r, c]: (r, c) for r in range(mat_f.shape[0]) for c in range(mat_f.shape[1])}

gc,gr = goal_f[mat_i[0,0]]
print(mat_i)
print(mat_f)
#print(num)
#print(gc)
#print(gr)
def subgoal_problem(inizio:np.ndarray, fine:np.ndarray) -> int:  
        """
        Euristica che assegna priorità alle colonne più a sinistra.
        Penalizza gli elementi fuori posto con pesi decrescenti per colonne successive.
        """
        goal_matrix = fine
        current_matrix = inizio
        rows, cols = current_matrix.shape
        score = 0
        for c in range(cols):
            for r in range(rows - 1, -1, -1):  # Dal basso verso l'alto
                if current_matrix[r][c] != goal_matrix[r][c]:
                    score += r + 1 # Penalità ponderata per colonna
                    print(score)
                elif current_matrix[r][c] == 0:
                    break
            if score > 0: break
        return score*np.sum((current_matrix!= 0) & (current_matrix != goal_matrix))

def posti_sbagliati_piu_giusti_sopra_piu_costo_sol(inizio:np.ndarray, fine:np.ndarray) -> int:
    corrente = inizio
    rows, cols = corrente.shape
    goal = fine
    goal_positions = {goal[r, c]: (r, c) for r in range(goal.shape[0]) for c in range(goal.shape[1])}
    errore = 0
    lista_col_goal = []

    for c in range(0, cols):
        for r in range(rows - 1, -1, -1):
            if corrente[r][c] != goal[r][c]:
                errore += 1
                for r_sopra in range(r - 1, -1, -1):  # controlliamo sopra
                    if corrente[r_sopra][c] != 0:
                        errore += 1
                    else:
                        break

                #calcoliamo il costo della soluzione
                gr, gc = goal_positions[corrente[r, c]]
                if gc not in lista_col_goal:
                    #print(lista_col_goal)
                    lista_col_goal.append(gc)
                    if gc == c: # sono nella stessa colonna
                        if (gr > r): #la soluzione si troverebbe sotto alla riga
                            errore += gr - r + 1# quindi il costo
                        else:
                            errore += r - gr + 1
                    else:
                        for r_sopra in range(gr, -1, -1):  # controlliamo sopra
                            if corrente[r_sopra][gc] != 0:
                                errore += 1
                            else:
                                break
                    break
    return errore

err = subgoal_problem(mat_i,mat_f)
print(err)