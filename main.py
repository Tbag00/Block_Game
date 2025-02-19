import cv2 as cv
from aima import astar_search
from prova import Mproblem, Matrice, execute
from puppolo import anima_matrice
from recognition_blocks import getStato
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Apri la finestra di dialogo per selezionare un'immagine
def upload_image() -> cv.Mat:
    file_path = filedialog.askopenfilename(title="Seleziona un'immagine", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    
    if file_path:
        # Carica l'immagine con OpenCV
        img = cv.imread(file_path)

        # Visualizza l'immagine in una finestra OpenCV
        cv.imshow("Immagine Caricata", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return img

if __name__ == "__main__":
    """
    immagine_iniziale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    #immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\puppalon.jpeg',cv.IMREAD_GRAYSCALE)"""
    # Crea la finestra principale
    root = tk.Tk()
    root.title("Block's World")

    # Crea il pulsante "Upload"
    upload_button = tk.Button(root, text="Upload Stato iniziale", command=upload_image)
    upload_button.pack(pady=20)

    # Crea il pulsante "Upload"
    upload_button = tk.Button(root, text="Upload Stato iniziale", command=upload_image)
    upload_button.pack(pady=20)

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