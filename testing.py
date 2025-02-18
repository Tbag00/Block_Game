from typing import final
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2

cv2.ocl.setUseOpenCL(False)
model: models.Sequential = models.load_model("recognition_numbers.keras")

img = cv2.imread('/home/tommaso/intelligenzaArtificiale/progetto/test_personali_numeri/2.jpeg', cv2.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
# colori da invertire se uso OTSU inverto colori perché dataset ha colori invertiti
#img = cv2.bitwise_not(img)

# miglioro contrasto
img = cv2.normalize(
    img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.ADAPTIVE_THRESH_GAUSSIAN_C) # da' immagini invertite
#(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# adatto input size
img = cv2.resize(img,(28,28))

# pixel<128 settati a 0, pixel >128 settati a 255, THRESH_BINARY per avere immagine binaria, THRESH_OTSU

print(img.shape)
print("tipo di img: ", type(img))
print(img)
"""final_im = np.expand_dims(img, axis=2)
print(final_im.shape)
print(type(final_im.shape))"""
print(model.predict(np.array([img])).argmax(axis=1))
plt.imshow(img)
plt.show()
#cv2.imshow('image',img)
#cv2.waitKey(0)

# closing all open windows
# cv2.destroyAllWindows()


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print(train_images[0].shape)
print("tipo di train image: ", type(train_images[0]))
print(train_images[0])
print(model.predict(np.array([train_images[0]])))
plt.imshow(train_images[0])
plt.show()
#guarda roi è la parte importante
'''import cv2
import pytesseract

# Carica l'immagine in scala di grigi
img = cv2.imread("immagine.png", cv2.IMREAD_GRAYSCALE)

# Trova i contorni
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  # Bounding box del rettangolo
    roi = img[y:y+h, x:x+w]  # Estrazione dell'area interna

    # OCR per leggere il numero dentro il rettangolo
    numero = pytesseract.image_to_string(roi, config='--psm 6')
    print(f"Numero trovato: {numero.strip()}")'''