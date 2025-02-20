# Libraries import
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import visualkeras
from contextlib import redirect_stdout
import cv2 as cv
from sklearn.metrics import accuracy_score

#from tensorflow.keras.callbacks import EarlyStopping

# inserire immagini dei numeri accettati da mnist

"""The filters in the first few layers are usually less abstract and typically emulates edge detectors, blob detectors etc. You generally don't want too many filters applied to the input layer as there is only so much information extractable from the raw input layer. Most of the filters will be redundant if you add too many. You can check this by pruning (decrease number of filters until your performance metrics degrade)

The kernel size determines how much of the image you want affecting the output of your convolution (the 'receptive field' of the kernel). It's been seen smaller kernels are generally better than larger ones (i.e go with 3x3 instead of 5x5, 7x7).

The Inception Architecture takes these decisions out of the hand of the modeller as it lumps filters of different kernel size together and let the model learn the best ones to use."""

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# tolgo label di numeri non necessari
filtro_train = (train_labels>0) & (train_labels <7)
filtro_test = (test_labels>0) & (test_labels <7)

plt.imshow(train_images[0])
plt.savefig("test_plot")

exit(0)
train_images = train_images[filtro_train]
train_labels = train_labels[filtro_train] - 1
test_images = test_images[filtro_test]
test_labels = test_labels[filtro_test] - 1

# normalizzo
train_images= train_images/255
test_images= test_images/255

labels = set(train_labels)

#definisco CNN
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # layers.Dropout(0.3),
    # layers.Conv2D(64, (5, 5), activation='relu'),
    # layers.MaxPooling2D((2,2)),
    # layers.Dense(64, activation='relu'),
    layers.Flatten(),
    layers.Dense(len(labels), activation='softmax')
])
# stampo e salvo model summary
model.summary()
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Fermo il training quando il validation loss raggiunge una soglia accettabile
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1, mode='min')

# Model training
# history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))
history = model.fit(train_images, train_labels, batch_size= 128, epochs=15, validation_split=0.1, callbacks=[early_stopping])

model.save("recognition_numbers.keras", overwrite= True)

# Degine a subplot grid 1x2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Plot for accuracy and val_accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)

# Plot for loss and val_loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.ylim([0.0, 2])
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig("Accuracy_Loss_CNN")
plt.show()