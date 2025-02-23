from tabnanny import verbose
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from recognition_blocks import getStato
from aima import astar_search
from prova import Mproblem, Matrice, execute
from puppolo import anima_matrice
import cv2 as cv
from modifica_matrice import edit_matrix
from PIL import Image, ImageTk

# Funzione per caricare l'immagine
def upload_image(is_iniziale: bool):
    file_path = filedialog.askopenfilename(
        title="Seleziona un'immagine", 
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
    )
    
    if file_path:
        # Carica l'immagine con OpenCV
        image = cv.imread(file_path)
        # Mostra l'immagine in una finestra separata
        show_image_in_window(image, is_iniziale)
        
        if is_iniziale:
            global immagine_iniziale
            immagine_iniziale = image
        else:
            global immagine_finale
            immagine_finale = image

# Funzione per mostrare l'immagine in una finestra separata usando matplotlib
def show_image_in_window(image, is_iniziale: bool):
    # Converte l'immagine da BGR a RGB per matplotlib
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Creazione della figura di matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')  # Rimuove gli assi
    title = "Stato Iniziale" if is_iniziale else "Stato Finale"
    plt.title(title)
    plt.show()

# bottone verbose: se "on" mostra tutti i passaggi della ricognizione
def verbose_toggle():
    global verbose
     
    # Determine is on or off
    if verbose:
        verbose_toggle.config(image = off)
        my_label.config(text = "Verbose non attivo", fg = "red")
        verbose = False
    else:
        verbose_toggle.config(image = on)
        my_label.config(text = "Verbose attivo", fg = "green")
        verbose = True

# Funzione per caricare l'immagine ridimensionata
def load_resized_image(image_path, width, height):
    img = Image.open(image_path)
    img_resized = img.resize((width, height), Image.Resampling.LANCZOS)

    # Converte l'immagine in formato compatibile con Tkinter
    img_tk = ImageTk.PhotoImage(img_resized)
    return img_tk

# Funzione di conferma
def confirm_images(verbose: bool):
    # controllo: entrambe immagini caricate
    if immagine_iniziale is None or immagine_finale is None:
        messagebox.showerror("Errore", "Devi caricare entrambe le immagini!")
        return

    # Prosegui con il calcolo e la visualizzazione
    matrice_iniziale = getStato(immagine_iniziale, verbose)
    matrice_finale = getStato(immagine_finale, verbose)

    print("matrice iniziale:")
    print(matrice_iniziale)
    print("matrice finale:")
    print(matrice_finale)

    matrice_iniziale = edit_matrix(matrice_iniziale)
    matrice_finale = edit_matrix(matrice_finale)

    print("matrice iniziale:")
    print(matrice_iniziale)
    print("matrice finale:")
    print(matrice_finale)

    problemazione = Mproblem(Matrice(matrice_iniziale), matrice_finale)
    
    soluzione2  = execute("A-Star euristica veloce", astar_search, problemazione, problemazione.posti_sbagliati_piu_giusti_sopra_piu_costo_sol)
    anima_matrice(matrice_iniziale, matrice_finale, soluzione2.solution())
    
    soluzione3  = execute("A-Star euristica relaxed pesata", astar_search, problemazione, problemazione.posti_sbagliati_piu_giusti_sopra)
    anima_matrice(matrice_iniziale, matrice_finale, soluzione3.solution())  
    
    soluzione1 = execute("A-Star euristica relaxed", astar_search, problemazione, problemazione.posti_sbagliati)
    anima_matrice(matrice_iniziale, matrice_finale, soluzione1.solution()) 
    messagebox.showinfo("Completato", "Le immagini sono state processate con successo!")

# Funzione di chiusura
def on_close():
    print("Window is closing!")
    root.quit()

# Funzione di uscita
def exit_program():
    root.quit()  # Chiude la finestra di Tkinter

# Codice principale
if __name__ == "__main__":
    immagine_iniziale = None
    immagine_finale = None
    verbose = False

    # Crea la finestra principale
    root = tk.Tk()
    root.title("Block's World")

    # Crea il frame per il layout
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=20)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Create Label
    my_label = Label(root, 
    text = "Verbose non attivo", 
    fg = "red")
    my_label.pack(pady = 20)
    
    on = load_resized_image("switch_png/on.png", 50, 50)  # Imposta la dimensione a 100x50 pixel
    off = load_resized_image("switch_png/off.png", 50, 50)  # Imposta la dimensione a 100x50 pixel

    # Crea i pulsanti
    upload_button1 = tk.Button(frame, text="Upload Stato Iniziale", command=lambda: upload_image(is_iniziale=True))
    upload_button1.pack(pady=10, padx=10)

    upload_button2 = tk.Button(frame, text="Upload Stato Finale", command=lambda: upload_image(is_iniziale=False))
    upload_button2.pack(pady=10, padx=10)

    # Pulsante di conferma per procedere
    confirm_button = tk.Button(frame, text="Conferma e Inizia", command=lambda: confirm_images(verbose))
    confirm_button.pack(pady=20)

    # Toggle verbose
    verbose_toggle = tk.Button(root, image = off, bd = 0, command = verbose_toggle)
    verbose_toggle.pack(pady=10, padx=10)

    # Pulsante di uscita
    exit_button = tk.Button(frame, text="Esci", command=exit_program)
    verbose_toggle.pack(pady=10, padx=10)

    # Avvia il loop principale
    root.mainloop()