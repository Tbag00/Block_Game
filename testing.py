from typing import final
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2 as cv

cv.ocl.setUseOpenCL(False)
model: models.Sequential = models.load_model("recognition_numbers.keras")

img = cv.imread('/home/tommaso/intelligenzaArtificiale/progetto/test_personali_numeri/2.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
# colori da invertire se uso OTSU inverto colori perché dataset ha colori invertiti
#img = cv.bitwise_not(img)

# miglioro contrasto
img = cv.normalize(
    img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
(thresh, img) = cv.threshold(img, 180, 255, cv.THRESH_BINARY | cv.ADAPTIVE_THRESH_GAUSSIAN_C) # da' immagini invertite
#(thresh, img) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# adatto input size
img = cv.resize(img,(28,28))

# pixel<128 settati a 0, pixel >128 settati a 255, THRESH_BINARY per avere immagine binaria, THRESH_OTSU

print(img.shape)
print("tipo di img: ", type(img))
print(img)
print(model.predict(np.array([img])).argmax(axis=1))
plt.imshow(img)
plt.show()
#cv.imshow('image',img)
#cv.waitKey(0)

# closing all open windows
# cv.destroyAllWindows()


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print(train_images[0].shape)
print("tipo di train image: ", type(train_images[0]))
print(train_images[0])
print(model.predict(np.array([train_images[0]])))
plt.imshow(train_images[0])
plt.show()
#guarda roi è la parte importante
'''import cv
import pytesseract

# Carica l'immagine in scala di grigi
img = cv.imread("immagine.png", cv.IMREAD_GRAYSCALE)

# Trova i contorni
contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)  # Bounding box del rettangolo

    # OCR per leggere il numero dentro il rettangolo
    numero = pytesseract.image_to_string(roi, config='--psm 6')
    print(f"Numero trovato: {numero.strip()}")'''