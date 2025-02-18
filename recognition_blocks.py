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
    plt.imshow(rect)
    plt.show()
    predictions = model.predict(np.array([rect]))#.argmax(axis=1)
    print(predictions)
    predictions[:, 7:] = -np.inf
    predictions[:, 0] = -np.inf
    print(predictions)
    return predictions.argmax(axis=1)


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
cv.destroyAllWindows()
