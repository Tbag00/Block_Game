from copy import deepcopy
from os import remove
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2 as cv

# controlla se rettangolo 1 contiene rettangolo 2
def inside(rectExt: dict, rectInt: dict) -> bool:
    if rectInt["x"] >= rectExt["x"] and rectInt["y"] >= rectExt["y"]:
        if rectInt["x"]+rectInt["w"] <= rectExt["x"]+ rectExt["w"]:
            if rectInt["y"]+rectInt["h"] <= rectExt["y"]+rectExt["h"]:
                return True
    else: 
        return False

# Input: immagine contenente solo numero
# Output: etichetta
def recon_number(rect: cv.Mat) -> int:
    # adatto input size
    rect = cv.bitwise_not(rect)
    rect = cv.resize(rect,(28,28))
    predictions = model.predict(np.array([rect]))#.argmax(axis=1)
    predictions[:, 7:] = -np.inf
    predictions[:, 0] = -np.inf
    return predictions.argmax(axis=1)


def apply_gravity(matrix) -> np.matrix:
    """Simula la gravità facendo cadere i numeri verso il basso."""
    rows, cols = matrix.shape
    for col in range(cols):
        non_zero_values = [matrix[row][col] for row in range(rows) if matrix[row][col] != 0]
        zero_count = rows - len(non_zero_values)
        matrix[:, col] = [0] * zero_count + non_zero_values  # Riempie con zeri sopra e numeri sotto
    return matrix

def costruisci_mat(rects: list) -> np.matrix:
    n = len(rects)
    """if n > 6:
        print("tvoppe")
        return"""
    mat = np.zeros((n, n), dtype=int)  # Crea una matrice quadrata di zeri
    
    col = 0
    row = 0
    inseriti = []
    for rect in rects:
        if rect not in inseriti:
            print("rect",rect["value"])
            row = 0
            for item in rects:
                if item not in inseriti:
                    if rect["x"]-rect["w"]/2.0 <= item["x"] <= rect["x"]+rect["w"]/2.0:
                        print("item",item["value"])
                        mat[row][col] = item["value"]
                        row += 1
                        inseriti.append(item)
                    else:
                        break
            col += 1
    print(mat)
    #mat[row,:] = sorted(mat[row,:], key= lambda val: )
    y_values = [rect["y"] for rect in rects]
    print(y_values)

    # Ordinare gli indici in base ai valori di "y"
    sorted_indices = np.argsort(y_values)
    print(sorted_indices)
    
    # Ordinare le righe della matrice in base all'ordine degli indici "y"
    mat = mat[sorted_indices, :]

    print(mat)
    apply_gravity(mat)
    print(mat)
    return mat
    """ for i in range(n):
        print("rect",rect["value"])
        row = 0
        mat[row][col] = rects[i]["value"]
        for j in range(n):
                if rects[i]["x"]-rects["w"]/2.0 <= rects[i]["x"] <= rect["x"]+rect["w"]/2.0:
                    print("item",item["value"])
                    mat[row][col] = item["value"]
                    row += 1
                else:
                    break
        col += 1
            """
# importo modello e immagine
model: models.Sequential = models.load_model("recognition_numbers.keras")
img = cv.imread('/home/tommaso/intelligenzaArtificiale/progetto/test_personali_blocks/ombroso.jpeg', cv.IMREAD_GRAYSCALE)
#img = cv.resize(img, (1024, 512))
assert img is not None, "file could not be read, check with os.path.exists()"
larghezza_img = img.shape[1]
altezza_img = img.shape[0]

"""
# miglioro contrasto
cv.imshow("test",img)
cv.waitKey(0)
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cv.imshow("test",img)
cv.waitKey(0)
img=clahe.apply(img)
blurred = cv.GaussianBlur(img, (21, 21), 0)
img = cv.absdiff(img, blurred)
img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)"""
#kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)
img = cv.normalize(
    img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
(thresh, img) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.ADAPTIVE_THRESH_GAUSSIAN_C)

# decommenta se usi caussiana poché inverte i colori quindi li inverto
img = cv.bitwise_not(img)

# lista contenente i rettangoli
# i rettangoli sono dizionari codificati dall' angolo in alto a sinistra (x,y) dalla base w e altezza h
rects = []
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x,y,w,h = cv.boundingRect(contour)
    ratio = w/float(h)

    # controllo se la forma e' rettangolare
    if 0.8 < ratio < 2.0:
        # approssimo rettangolo
        epsilon = 0.14*cv.arcLength(contour,True)
        approx = cv.approxPolyDP(contour,epsilon,True)

        # controllo che contour sia rettangolo
        if len(approx) == 4:
            x,y,w,h = cv.boundingRect(approx)

            # salvo rettangoli con dimensioni limitate
            if w > (larghezza_img / 12) and h > (altezza_img / 12) and w < (larghezza_img / 4) and h < (altezza_img / 4):
                rects.append({
                    "x":x, "y":y, "w":w, "h":h, "value":0
                })

# ordino i rettangoli da sinistra a destra
rects = sorted(rects, key= lambda r:r["x"])

# escludo rettangoli esterni
for rectExt in rects:
    for rectInt in rects:
        if inside(rectExt, rectInt):
            print("removed triangle: %s" %rectExt["x"])
            rects.remove(rectExt)

for rect in rects:
    rect_img = img[rect["y"]+10:rect["y"]+rect["h"]-10, rect["x"]+10:rect["x"]+rect["w"]-10]
    #cv.imshow("numero", rect_img)
    #cv.waitKey(0)
    rect["value"] = recon_number(rect_img)
    print(rect["value"])

costruisci_mat(rects)
cv.destroyAllWindows()

#drinkodice

'''import numpy as np

def costruisci_mat(rects, n):
    """Costruisce una matrice nxn ordinata per x (colonna) e y (altezza)."""

    tolleranza_x = 10  # Permette di raggruppare x simili
    colonne = {}

    # Raggruppamento per x
    for rect in rects:
        x = rect["x"]
        trovato = False

        for key in colonne.keys():
            if abs(key - x) < tolleranza_x:
                colonne[key].append(rect)
                trovato = True
                break

        if not trovato:
            colonne[x] = [rect]

    # Ordina le colonne per x
    colonne_ordinate = sorted(colonne.items(), key=lambda item: item[0])

    # Costruisce la matrice
    matrice = []
    for _, col in colonne_ordinate:
        col.sort(key=lambda rect: rect["y"])  # Ordina per y (dall'alto in basso)
        matrice.append([rect["value"] for rect in col])

    # Rende la matrice n x n
    while len(matrice) < n:  # Se ci sono meno di n colonne, aggiunge colonne vuote
        matrice.append([])

    for col in matrice:
        while len(col) < n:  # Se una colonna ha meno di n righe, riempie con 0
            col.append(0)

    # Converte in array NumPy per facilitare l'uso
    mat_np = np.array(matrice).T  # Trasposta per ottenere la forma corretta

    print(mat_np)
    return mat_np
'''
