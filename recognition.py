# Libraries import
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images_raw, train_labels), (test_images_raw, test_labels) = mnist.load_data()
# normalizzo
(train_images_raw)