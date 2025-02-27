from copy import deepcopy
import copy
from os import remove
import re
import numpy as np
from pygame import ver
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2 as cv

# importo modello
model: models.Sequential = models.load_model("recognition_numbers.keras")

# schiarisce immagine
def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

# converte immagine con blocchi numerati in stato (matrice)
def getStato(original_img: cv.Mat, verbose: bool) -> np.array:
    assert original_img is not None, "file could not be read, check with os.path.exists()"

    # dati immagine
    larghezza_img = original_img.shape[1]
    altezza_img = original_img.shape[0]
    print("larghezza:", larghezza_img)
    print("altezza:", altezza_img)
    
    # correzione immagine
    img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY) # scala grigi
    img = adjust_gamma(img, gamma=1.7) # schiarisce immagine
    
    # normalizza
    img = cv.normalize(
        img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    (thresh, img) = cv.threshold(img, 180, 255, cv.THRESH_BINARY | cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    
    # controllo immagine elaborata
    if verbose:
        cv.imshow("immagine elaborata", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # decommenta se usi gaussiana poché inverte i colori
    img = cv.bitwise_not(img)

    # lista contenente i rettangoli
    # i rettangoli sono dizionari codificati dall' angolo in alto a sinistra (x,y) dalla base w e altezza h
    rects = []
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        ratio = w / float(h)

        # controllo se la forma e' rettangolare
        if 0.8 < ratio < 2.0:

            # approssimo rettangolo
            epsilon = 0.01 * cv.arcLength(contour, True) # piu' e' alto meno restrittivo e' nell'approssimazione delle righe dritte
            approx = cv.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv.boundingRect(approx)

            # stampo dati figura
            print("x: %s, w: %s, y: %s, h: %s" %(x, w, y, h))
            print("lati:", len(approx))
            
            img_contoured: cv.Mat # usata solo se verbose

            # controllo che contour sia rettangolo
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(approx)
                
                # salvo rettangoli con dimensioni limitate
                if w > (larghezza_img / 16.0) and h > (altezza_img / 16.0) and w < (larghezza_img / 4.0) and h < (altezza_img / 4.0):
                    if verbose: 
                        img_contoured = img.copy()
                        cv.rectangle(img_contoured, (x, y), (x + w, y + h), (255, 0, 0), 10)
                        cv.imshow("rectangle", img_contoured)
                        cv.waitKey(0)

                    rects.append({
                        "x": x, "y": y, "w": w, "h": h, "value": 0
                    })
    cv.destroyAllWindows()

    # ordino i rettangoli da sinistra a destra
    rects = sorted(rects, key=lambda r: r["x"])

    # escludo rettangoli esterni
    inner_recs = []
    for rectExt in rects:
        exterior = False
        for rectInt in rects:
            if inside(rectExt, rectInt) and not rectExt is rectInt:
                print("removed rectangle: %s" % rectExt["x"])
                exterior = True
        if exterior == False:
            inner_recs.append(rectExt)

    for rect in inner_recs:
        rect_img = original_img[rect["y"] + 12 :rect["y"] + rect["h"] - 12, rect["x"] + 12 :rect["x"] + rect["w"] - 12]
        if verbose:
            cv.imshow("number inside box",rect_img)
            cv.waitKey(0)

        rect["value"] = recon_number(rect_img, verbose)
        print(rect["value"])
    print("ho rilevato %s blocchi" %len(inner_recs))
    cv.destroyAllWindows()
    return costruisci_mat(inner_recs)

# controlla se rettangolo 1 contiene rettangolo 2
def inside(rectExt: dict, rectInt: dict) -> bool:
    if rectInt["x"] > rectExt["x"] and rectInt["y"] > rectExt["y"]:
        if rectInt["x"] + rectInt["w"] < rectExt["x"] + rectExt["w"]:
            if rectInt["y"] + rectInt["h"] < rectExt["y"] + rectExt["h"]:
                return True
    else:
        return False


# Input: immagine contenente solo numero
# Output: etichetta
def recon_number(rect: cv.Mat, verbose: bool) -> int:
    # adatto input size
    rect = cv.bitwise_not(rect)
    rect = cv.resize(rect, (28, 28))
    rect = cv.cvtColor(rect, cv.COLOR_BGR2GRAY) # scala grigi
    # rect = cv.normalize(
    #     rect, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    (thresh1, rect) = cv.threshold(rect, 127, 255, cv.THRESH_TOZERO)
    
    if verbose:
        cv.imshow("immagine che riconosce", rect)
        cv.waitKey(0)
    rect = rect/255

    #print(rect)
    predictions = model.predict(np.array([rect])) 
    return predictions.argmax(axis=1) + 1


# Simula la gravità facendo cadere i numeri verso il basso.
def apply_gravity(matrix) -> np.matrix:
    rows, cols = matrix.shape
    for col in range(cols):
        non_zero_values = [matrix[row][col] for row in range(rows) if matrix[row][col] != 0]
        zero_count = rows - len(non_zero_values)
        matrix[:, col] = [0] * zero_count + non_zero_values  # Riempie con zeri sopra e numeri sotto
    return matrix

"""
def costruisci_mat(rects: list) -> np.matrix:
    n = len(rects)
    mat = np.zeros((n, n, 2), dtype=int)  # Crea una matrice quadrata di zeri

    col = 0
    row = 0
    inseriti = []
    for rect in rects:
        if rect not in inseriti:
            print("rect", rect["value"])
            row = 0
            for item in rects:
                if item not in inseriti:
                    if rect["x"] - rect["w"] / 2.0 <= item["x"] <= rect["x"] + rect["w"] / 2.0:
                        print("item", item["value"])
                        mat[row][col][0] = item["value"]
                        mat[row][col][1] = item["y"]
                        row += 1
                        inseriti.append(item)
                    else:
                        break
            col += 1
    # ordine crescente
    mat = np.sort(mat, axis=2)
    res = mat[:, :, 0]
    res = res[::-1]  # ordine decrescente

    apply_gravity(res)
    return res
"""
def costruisci_mat(rects: list) -> np.matrix:
    # Collezione per x raggruppa i rettangoli in colonne, le colonne hanno x molto vicina.
    collezione_per_x = []
    
    # Raggruppa i rettangoli che appartengono allo stesso intervallo di x
    for rect in rects:
        trovato = False
        for col in collezione_per_x:
            # Verifica se il rettangolo appartiene a questa colonna
            if (rect["x"] - rect["w"]/2 <= col[0]["x"] + col[0]["w"]/2) and (rect["x"] + rect["w"]/2 >= col[0]["x"] - col[0]["w"]/2):
                col.append(rect)
                trovato = True
                break
        
        # Se non trovato, crea una nuova colonna per il nuovo gruppo di x
        if not trovato:
            collezione_per_x.append([rect])
    
    # Prepara la matrice finale con il numero massimo di rettangoli per colonna
    # max_rects_in_col = max(len(col) for col in collezione_per_x)
    # mat = np.zeros((max_rects_in_col, len(collezione_per_x)), dtype=int)
    n = len(rects)
    mat = np.zeros((n, n), dtype=int)  # Crea una matrice quadrata di zeri
    
    # Ordina i rettangoli per ogni gruppo in base a y e inseriscili nella matrice
    for col_idx, col in enumerate(collezione_per_x):
        # Ordina per y
        col_sorted = sorted(col, key=lambda x: x["y"])
        
        # Inserisci nella colonna della matrice
        for row_idx, rect in enumerate(col_sorted):
            mat[row_idx, col_idx] = rect["value"]

    apply_gravity(mat)
    
    return mat