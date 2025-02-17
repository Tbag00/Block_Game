import numpy as np
import tensorflow as tf
import keras
import cv2

cv2.ocl.setUseOpenCL(False)
model = keras.models.load_model("recognition_numbers.keras")
img = cv2.imread('/home/tommaso/intelligenzaArtificiale/progetto/test_personali_numeri/1.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(28,28))

# pixel<128 settati a 0, pixel >128 settati a 255, THRESH_BINARY per avere immagine binaria, THRESH_OTSU
normalized_image = cv2.normalize(
    img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
(thresh, im_bw) = cv2.threshold(normalized_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

model.predi
#cv2.imshow('image',im_bw)
#cv2.waitKey(0)

# closing all open windows
# cv2.destroyAllWindows()
