from copy import deepcopy
import copy
from os import remove
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2 as cv

# importo modello
model: models.Sequential = models.load_model("recognition_numbers.keras")
def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def getStato(img: cv.Mat) -> np.array:
    assert img is not None, "file could not be read, check with os.path.exists()"

    # dati immagine
    larghezza_img = img.shape[1]
    altezza_img = img.shape[0]
    print("larghezza:", larghezza_img)
    print("altezza:", altezza_img)
    
    # correzione immagine
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # scala grigi
    img = adjust_gamma(img, gamma=1.7) # schiarisce immagine
    # inspessisce bordi
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3)) 
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)
    # normalizza
    img = cv.normalize(
        img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    (thresh, img) = cv.threshold(img, 180, 255, cv.THRESH_BINARY | cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    
    # controllo immagine elaborata
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
            
            """ # controllo se non funziona
            cv.drawContours(img, [contour], -1, (0,255,255), 20)
            cv.drawContours(img, [approx], -1, (0,0,255), 10)
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
            cv.imshow("test2", img)
            cv.waitKey(0)
            """

            # controllo che contour sia rettangolo
            if len(approx) == 4:
                x, y, w, h = cv.boundingRect(approx)
                
                """ # controllo se non funziona
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
                cv.imshow("test2", img)
                cv.waitKey(0)
                """
                # salvo rettangoli con dimensioni limitate
                if w > (larghezza_img / 16.0) and h > (altezza_img / 16.0) and w < (larghezza_img / 4.0) and h < (altezza_img / 4.0):
                    """ # mostro triangoli
                    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
                    cv.imshow("test2", img)
                    cv.waitKey(0)
                    """
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
        rect_img = img[rect["y"] + 10:rect["y"] + rect["h"] - 10, rect["x"] + 10:rect["x"] + rect["w"] - 10]
        rect["value"] = recon_number(rect_img)
        print(rect["value"])
    print("ho rilevato %s blocchi" %len(inner_recs))
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
def recon_number(rect: cv.Mat) -> int:
    # adatto input size
    rect = cv.bitwise_not(rect)
    rect = cv.resize(rect, (28, 28))
    predictions = model.predict(np.array([rect]))  # .argmax(axis=1)
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
    res = res[::-1, :]  # ordine decrescente

    apply_gravity(res)
    print(res)
    return res


'''  # img = cv.imread('/home/tommaso/intelligenzaArtificiale/progetto/test_personali_blocks/img_ultrahd.jpeg', cv.IMREAD_GRAYSCALE)
# img = cv.resize(img, (1024, 512))
assert img is not None, "file could not be read, check with os.path.exists()"
larghezza_img = img.shape[1]
altezza_img = img.shape[0]
'''

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
'''# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
# img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)
img = cv.normalize(
    img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
(thresh, img) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.ADAPTIVE_THRESH_GAUSSIAN_C)

# decommenta se usi caussiana poché inverte i colori quindi li inverto
img = cv.bitwise_not(img)

# lista contenente i rettangoli
# i rettangoli sono dizionari codificati dall' angolo in alto a sinistra (x,y) dalla base w e altezza h
rects = []
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("test", img)
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    ratio = w / float(h)

    # controllo se la forma e' rettangolare
    if 0.8 < ratio < 2.0:
        # approssimo rettangolo
        epsilon = 0.14 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # controllo che contour sia rettangolo
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)

            # salvo rettangoli con dimensioni limitate
            if w > (larghezza_img / 12) and h > (altezza_img / 12) and w < (larghezza_img / 4) and h < (
                    altezza_img / 4):
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
                cv.imshow("test2", img)
                cv.waitKey(0)
                rects.append({
                    "x": x, "y": y, "w": w, "h": h, "value": 0
                })

# ordino i rettangoli da sinistra a destra
rects = sorted(rects, key=lambda r: r["x"])

# escludo rettangoli esterni
inner_recs = copy.deepcopy(rects)
for rectExt in rects:
    exterior = False
    for rectInt in rects:
        if inside(rectExt, rectInt) and not rectExt is rectInt:
            print("removed rectangle: %s" % rectExt["x"])
            exterior = True
    if exterior == False:
        inner_recs.append(rectExt)

for rect in inner_recs:
    rect_img = img[rect["y"] + 10:rect["y"] + rect["h"] - 10, rect["x"] + 10:rect["x"] + rect["w"] - 10]
    # cv.imshow("numero", rect_img)
    # cv.waitKey(0)
    rect["value"] = recon_number(rect_img)
    print(rect["value"])

costruisci_mat(rects)
cv.destroyAllWindows()'''