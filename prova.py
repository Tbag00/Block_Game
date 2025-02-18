from collections.abc import Callable
from collections import deque
from typing import List, Tuple
import numpy as np
import random
from aima import Problem, Node, memoize, PriorityQueue, GraphProblem
from aima import depth_first_graph_search, breadth_first_graph_search, iterative_deepening_search, \
     depth_limited_search, astar_search
from collections.abc import Callable
import time
from dataclasses import dataclass
from puppolo import anima_matrice

"""HA SEGNATO L'INTER"""
BLUE = "\033[34;1m"
RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"


def apply_gravity(matrix):
    """Simula la gravità facendo cadere i numeri verso il basso."""
    rows, cols = matrix.shape
    for col in range(cols):
        non_zero_values = [matrix[row][col] for row in range(rows) if matrix[row][col] != 0]
        zero_count = rows - len(non_zero_values)
        matrix[:, col] = [0] * zero_count + non_zero_values  # Riempie con zeri sopra e numeri sotto
    return matrix


def generate_matrix():
    num_elements = 6  # Numero casuale di elementi unici tra 1 e 6
    numbers = random.sample(range(1, num_elements + 1), num_elements)  # Genera numeri unici

    matrix = np.zeros((num_elements, num_elements), dtype=int)  # Crea una matrice quadrata di zeri

    positions = random.sample(range(num_elements * num_elements), num_elements)  # Seleziona posizioni casuali
    for pos, num in zip(positions, numbers):
        row, col = divmod(pos, num_elements)
        matrix[row][col] = num  # Inserisce un numero unico tra 1 e 6

    return apply_gravity(matrix)


matrix_i = generate_matrix()
matrix_f = generate_matrix()

print(matrix_i)  #16 a 9
print(matrix_f)


def prova(state: np.ndarray) -> list[tuple[int, int]]:
    colonne = state.shape[1]
    righe = state.shape[0]
    print(righe)
    azioni_possibili = []
    for i in range(colonne):
        if state[righe - 1][i] != 0:
            for j in range(colonne):
                if i != j:
                    if state[0][j] == 0:
                        azioni_possibili.append((i, j))
                    # azioni_possibili.append(j)
    return azioni_possibili


pisello = prova(matrix_i)
print(pisello)

'''def nata_prova(state:np.ndarray)->None:
    righe_p = 0 #righe di partenza 
    righe_a = state.shape[0]-1 #righe di arrivo per le colonne di arrivo
    while state[righe_p][pisello[0][0]] == 0:
        righe_p += 1
    while state[righe_a][pisello[1][0]] != 0:
        righe_a -= 1
    state[righe_a][pisello[1][0]] = state[righe_p][pisello[0][0]]
    state[righe_p][pisello[0][0]] = 0
    return state
matrix_i = nata_prova(matrix_i)
print(matrix_i)
'''


@dataclass  #decoratore
class Result:
    result: Node
    nodes_generated: int
    paths_explored: int
    nodes_left_in_frontier: int


def breadth_first_graph_search1(problem: Problem, f: Callable) -> Result:
    print(pisello)
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    counter = 1
    while frontier:
        node = frontier.pop()

        # print(f"Exploring: \n{node.state.m_corrente}")  # Add this to see the states being explored
        #culo = np.array_equal(node.state.m_corrente,node.state.goal)
        # print(culo)

        if problem.goal_test(node.state):
            #print(node, counter, len(explored), len(frontier))
            return Result(node, counter, len(explored), len(frontier))
        explored.add(node.state)
        for child in node.expand(problem):
            counter += 1
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return Result(None, counter, len(explored), 0)


def execute(name: str, algorithm: Callable, problem: Problem, *args, **kwargs) -> None:
    print(f"{RED}{name}{RESET}\n")
    #uso perf_counter per contare "meglio" il tempo hardware al massimo
    start = time.perf_counter()
    sol = algorithm(problem, *args, **kwargs)
    end = time.perf_counter()
    if problem.goal is not None:
        print(f"\n{GREEN}PROBLEM:\n{RESET} {problem.initial.m_corrente} \n  |\n  v\n {problem.goal}")
    if isinstance(sol, Result):
        print(f"{GREEN}Total nodes generated:{RESET} {sol.nodes_generated}")
        print(f"{GREEN}Paths explored:{RESET} {sol.paths_explored}")
        print(f"{GREEN}Nodes left in frontier:{RESET} {sol.nodes_left_in_frontier}")
        sol = sol.result
    print(f"{GREEN}Result:{RESET} {sol.solution() if sol is not None else '---'}")
    if isinstance(sol, Node):
        print(f"{GREEN}Path Cost:{RESET} {sol.path_cost}")
        print(f"{GREEN}Path Length:{RESET} {sol.depth}")
    print(f"{GREEN}Time:{RESET} {end - start} s")
    anima_matrice(problem.initial.m_corrente, problem.goal, sol.solution())


class Matrice:
    def __lt__(self, other):
        return np.sum(self.m_corrente) < np.sum(other.m_corrente)

    def __init__(self, m_corrente: np.ndarray):
        self.m_corrente = m_corrente

    #eq serve a poter effettivamente confrontare due oggetti altrimenti python controlla che siano lo stesso oggetto con lo stesso indirizzo di memoria
    def __eq__(self, altra_m):
        if isinstance(altra_m, Matrice):
            return np.array_equal(self.m_corrente, altra_m.m_corrente)
        return False
        #hash invece mi da un valore numerico(int) che permette appunto identificare gli oggetti e in caso
        #di confronti appunto compara l'hash

    def __hash__(self):
        return hash(self.m_corrente.tobytes())


class Mproblem(Problem):
    def __init__(self, initial: Matrice, goal: Matrice) -> None:

        super().__init__(initial, goal)

    def actions(self, state: Matrice) -> list[tuple[int, int]]:
        colonne = state.m_corrente.shape[1]
        righe = state.m_corrente.shape[0] - 1
        azioni_possibili = []
        for i in range(colonne):
            if state.m_corrente[righe][i] != 0:
                for j in range(colonne):
                    if i != j:
                        if state.m_corrente[0][j] == 0:
                            azioni_possibili.append((i, j))
        return azioni_possibili

    def result(self, state: Matrice, action: tuple[int, int]) -> Matrice:
        new_matrice = np.copy(state.m_corrente)
        righe_p = 0  #righe di partenza
        righe_a = new_matrice.shape[0] - 1  #righe di arrivo per le colonne di arrivo
        while new_matrice[righe_p][action[0]] == 0:
            righe_p += 1
        while new_matrice[righe_a][action[1]] != 0:
            righe_a -= 1
        if righe_a >= 0:
            new_matrice[righe_a][action[1]] = new_matrice[righe_p][action[0]]  #modificare gli indici di action se da problemi perchè potrebbe volere un numero al posto di una tupla vedi sopra
            new_matrice[righe_p][action[0]] = 0
        return Matrice(new_matrice)

    def goal_test(self, state: Matrice) -> bool:
        return np.array_equal(state.m_corrente, self.goal)

    def relaxed_problem(problem, node) -> int:
        return np.sum(node.state.m_corrente != problem.goal)

    def weighted_relax(problem, node) -> int:
        stato_corrente = node.state.m_corrente
        goal = problem.goal
        rows, cols = stato_corrente.shape

        errore_totale = 0

        for r in range(rows):
            for c in range(cols):
                valore = stato_corrente[r, c]
                goal_valore = goal[r, c]
                if valore != goal_valore:  # se il valore è sbagliato il numero di azioni per metterlo al posto giusto dipendera dai blocchi sopra
                    errore_totale += 1
                    for r_sopra in range(r - 1, rows):  # controlliamo sopra
                        if stato_corrente[r_sopra][c] != 0:
                            errore_totale += 1
                        else:
                            break
                else:
                    # Il blocco è nella posizione giusta, ma ha sotto elementi sbagliati?
                    for r_sotto in range(r + 1, rows):  # Controlliamo sotto
                        if stato_corrente[r_sotto, c] != goal[r_sotto, c]:
                            errore_totale += rows - r_sotto
                            break  # Basta un errore sotto per contare il blocco come problematico

        return errore_totale

    def subgoal_problem(problem, node) -> int:
        """
        Euristica che assegna priorità alle colonne più a sinistra.
        Penalizza gli elementi fuori posto con pesi decrescenti per colonne successive.
        """
        goal_matrix = problem.goal
        current_matrix = node.state.m_corrente
        rows, cols = current_matrix.shape
        score = 0

        for col in range(cols):
            col_weight = (cols - col)  # Peso più alto per le prime colonne
            for row in range(rows - 1, -1, -1):  # Dal basso verso l'alto
                if current_matrix[row][col] != goal_matrix[row][col]:
                    score += col_weight  # Penalità ponderata per colonna

        return score

    def heavy_weighted_subgoal(problem, node) -> int:
        goal_matrix = problem.goal
        stato_corrente = node.state.m_corrente
        rows, cols = stato_corrente.shape
        score = 0

        for c in range(cols):
            col_weight = (cols - c)  # Peso più alto per le prime colonne
            for r in range(rows - 1, -1, -1):  # Dal basso verso l'alto
                if stato_corrente[r][c] != goal_matrix[r][c]:
                    peso = 1
                    for r_sopra in range(r - 1, rows):  # controlliamo sopra
                        if stato_corrente[r_sopra][c] != 0:
                            peso += 1
                        else:
                            score += col_weight * peso  # Penalità ponderata per colonna
                            break
                else:
                    # Il blocco è nella posizione giusta, ma ha sotto elementi sbagliati?
                    for r_sotto in range(r + 1, rows):  # Controlliamo sotto
                        if stato_corrente[r_sotto, c] != goal_matrix[r_sotto, c]:
                            score += (rows - r_sotto) * col_weight
                            break  # Basta un errore sotto per contare il blocco come problematico

        return score

    def weighted_subgoal(problem, node) -> int:
        goal_matrix = problem.goal
        stato_corrente = node.state.m_corrente
        rows, cols = stato_corrente.shape
        score = 0

        for c in range(cols):
            col_weight = (cols - c)  # Peso più alto per le prime colonne
            for r in range(rows - 1, -1, -1):  # Dal basso verso l'alto
                if stato_corrente[r][c] != goal_matrix[r][c]:
                    score += col_weight  # Penalità ponderata per colonna
                else:
                    # Il blocco è nella posizione giusta, ma ha sotto elementi sbagliati?
                    for r_sotto in range(r + 1, rows):  # Controlliamo sotto
                        if stato_corrente[r_sotto, c] != goal_matrix[r_sotto, c]:
                            score += 1
                            break  # Basta un errore sotto per contare il blocco come problematico

        return score

    def manhattan_distance(problem, node) -> int:
        stato_corrente = node.state.m_corrente
        goal = problem.goal

        # Creiamo una mappa delle posizioni dei valori nel goal
        goal_positions = {goal[r, c]: (r, c) for r in range(goal.shape[0]) for c in range(goal.shape[1])}

        distanza = 0
        for r in range(stato_corrente.shape[0]):
            for c in range(stato_corrente.shape[1]):
                valore = stato_corrente[r, c]
                if valore != 0:  # Evitiamo di cercare gli 0
                    gr, gc = goal_positions[valore]
                    distanza += abs(gr - r) + abs(gc - c)  # Distanza di Manhattan

        return distanza


matrice_inizio = Matrice(matrix_i)
problemazione = Mproblem(matrice_inizio, matrix_f)
execute("A-Star euristica subgoal pesata", astar_search, problemazione, problemazione.weighted_subgoal)
#execute("A-Star euristica subgoal pesata++", astar_search, problemazione, problemazione.heavy_weighted_subgoal)

