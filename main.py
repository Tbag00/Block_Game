from os import wait
import cv2 as cv
from aima import astar_search
from prova import Mproblem, Matrice, execute
from puppolo import anima_matrice
from recognition_blocks import getStato
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Apri la finestra di dialogo per selezionare un'immagine
def upload_image(is_iniziale: bool):
    file_path = filedialog.askopenfilename(
        title="Seleziona un'immagine", 
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
    )
    
    
    if file_path:
        # Carica l'immagine con OpenCV
        if is_iniziale:
            global immagine_iniziale
            immagine_iniziale = cv.imread(file_path)
        else:
            global immagine_finale
            immagine_finale = cv.imread(file_path)

# Definisci la funzione on_close
def on_close():
    print("Window is closing!")
    root.quit()  # Esce dal mainloop e chiude la finestra

if __name__ == "__main__":
    """
    immagine_iniziale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    #immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\iniziale.jpeg',cv.IMREAD_GRAYSCALE)
    immagine_finale = cv.imread('C:\\Users\\drink\\Downloads\\puppalon.jpeg',cv.IMREAD_GRAYSCALE)"""
    '''immagine_iniziale: cv.Mat = None
    immagine_finale: cv.Mat = None
    # Crea la finestra principale
    root = tk.Tk()
    root.title("Block's World")

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Crea il pulsante "Upload"
    upload_button = tk.Button(root, text="Upload Stato iniziale", command=lambda:upload_image(is_iniziale= True))
    upload_button.pack(pady=20)

    # Crea il pulsante "Upload"
    upload_button = tk.Button(root, text="Upload Stato finale", command=lambda:upload_image(is_iniziale= False))
    upload_button.pack(pady=20)

    # Start the main event loop
    root.mainloop()

    cv.imshow("iniziale",immagine_iniziale)
    cv.waitKey(0)
    cv.imshow("finale", immagine_finale)
    cv.waitKey(0)
    cv.destroyAllWindows()'''
    immagine_iniziale = cv.imread("/home/tommaso/Downloads/iniziale.jpeg")
    immagine_finale = cv.imread("/home/tommaso/Downloads/finale.jpeg")
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