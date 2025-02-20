import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import visualkeras
from contextlib import redirect_stdout

def visualize_features_map(im: np.ndarray, layer_index: int):

    # NOTE: You can get the model by its name, but consider that the names assigned change if you re-run the code. It's better to select the layer using the list index
    model_v = keras.Model(inputs = model.inputs[0], outputs = model.layers[layer_index].output)
    model_v.summary()

    # Get the feature maps
    feature_maps = model_v.predict(np.array([im]), verbose=False)[0]

    # Print the shape of feature_maps
    print("Feature maps shape:", feature_maps.shape)

    # Predict class name
    p = model.predict(np.array([im]), verbose=False).argmax(axis=1)
    print("Image class name:", p)

    # Show the image for which we want to compute the feature maps and its class
    plt.imshow(im)
    plt.show()

    # Show the feature map corresponding to a given filter as an image
    fmap=feature_maps[:,:,5]

    plt.imshow(fmap, cmap="gray")
    plt.show()

    # Show all the feature maps
    fig  = plt.figure(figsize=(10, 10))
    for i in range(feature_maps.shape[2]):
        sub = fig.add_subplot(8, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        sub.imshow(feature_maps[:,:,i], cmap = "gray")

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
batch_size = 128
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Fermo il training quando il validation loss raggiunge una soglia accettabile
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1, mode='min')

# Model training
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=15, validation_split=0.1, callbacks=[early_stopping])

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