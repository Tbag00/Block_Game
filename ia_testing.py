import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import datasets, models, layers, callbacks
import matplotlib.pyplot as plt
import cv2 as cv

# Degine a subplot grid 1x2
model = models.load_model("")
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


def visualize_features_map(im: np.ndarray, layer_index: int):

    # NOTE: You can get the model by its name, but consider that the names assigned change if you re-run the code. It's better to select the layer using the list index
    model_v = keras.Model(inputs = model.inputs[0], outputs = model.layers[layer_index].output)
    model_v.summary()

    # Get the feature maps
    feature_maps = model_v.predict(np.array([im]), verbose=False)[0]

    # Print the shape of feature_maps
    print("Feature maps shape:", feature_maps.shape)

    # Predict class name
    p = model.predict(np.array([im]), verbose=False)
    print("Image class name:", class_names[np.argmax(p)])

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