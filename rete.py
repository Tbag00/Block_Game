import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import visualkeras
from contextlib import redirect_stdout

# scarico dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(test_images.shape)
print(test_images.shape[0])

# rendo flat la matrice che rappresenta l' immagine
train_images = train_images.reshape(train_images.shape[0], 28*28)
test_images = test_images.reshape(test_images.shape[0], 28*28)

print(test_images.shape)
print(test_images.shape[0])
# normalizzo
train_images= train_images/255
test_images= test_images/255

model = models.Sequential([
    layers.Input(shape=(28*28,)),   # shape unidimensionale in una rete neurale semplice
    layers.Dense(512, activation= "relu"),
    layers.Dense(256, activation = "relu"),
    layers.Dense(10, activation= "softmax")
])

# stampo e salvo model summary
model.summary()
with open('rete_summary.txt', 'w') as f:
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
plt.savefig("Accuracy_Loss_Rete")
plt.show()

model.save("rete.keras", overwrite= True)