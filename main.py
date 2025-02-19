import cv2 as cv
from aima import astar_search
from prova import Mproblem, Matrice, execute
from puppolo import anima_matrice
from recognition_blocks import getStato
import numpy as np

if __name__ == "__main__":
    immagine_iniziale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    #immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\puppalon.jpeg',cv.IMREAD_GRAYSCALE)
    matrice_iniziale = getStato(immagine_iniziale)
    matrice_finale = getStato(immagine_finale)
    print(matrice_iniziale)
    print("matrice finale:")
    print(matrice_finale)
    '''    if matrice_iniziale.shape[0] != matrice_finale.shape[0]:
        if matrice_iniziale.shape[0] > matrice_finale.shape[0]:
            nuova_dimensione =
            matrice_quadrata = np.zeros((nuova_dimensione, nuova_dimensione), dtype=int)
            
            
            
            matrice_finale = np.hstack((matrice_finale, np.zeros((matrice_finale.shape[0], matrice_iniziale.shape[0] - matrice_finale.shape[0]), dtype=int)))
        else:
            matrice_iniziale = np.hstack((matrice_iniziale, np.zeros((matrice_iniziale.shape[0], matrice_finale.shape[0] - matrice_iniziale.shape[0]), dtype=int)))'''
    problemazione = Mproblem(Matrice(matrice_iniziale),matrice_finale)
    soluzione1 = execute("A-Star euristica subgoal pesata", astar_search, problemazione, problemazione.weighted_subgoal)
    anima_matrice(matrice_iniziale, matrice_finale, soluzione1)
    soluzione2  = execute("A-Star euristica ammissibile", astar_search, problemazione, problemazione.relaxed_problem)
    anima_matrice(matrice_iniziale, matrice_finale, soluzione2)