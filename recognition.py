# Libraries import
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import visualkeras
from keras.callbacks import EarlyStopping


"""The filters in the first few layers are usually less abstract and typically emulates edge detectors, blob detectors etc. You generally don't want too many filters applied to the input layer as there is only so much information extractable from the raw input layer. Most of the filters will be redundant if you add too many. You can check this by pruning (decrease number of filters until your performance metrics degrade)

The kernel size determines how much of the image you want affecting the output of your convolution (the 'receptive field' of the kernel). It's been seen smaller kernels are generally better than larger ones (i.e go with 3x3 instead of 5x5, 7x7).

The Inception Architecture takes these decisions out of the hand of the modeller as it lumps filters of different kernel size together and let the model learn the best ones to use."""

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# stampa shape training images
print(train_images.shape)
# stampa alcune training images
""" plt.figure(figsize=(10,10))
for i,image in enumerate(train_images[0:25]):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
plt.show() """

# normalizzo
train_images= train_images/255
test_images=test_images/255

#definisco CNN
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Fermo il training quando il validation loss raggiunge una soglia accettabile
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=1, mode='min')

# Model training
#history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))
history = model.fit(train_images, train_labels, epochs=15, validation_split=0.1, callbacks=[early_stopping])

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
plt.show()

model.save("recognition_numbers.keras")