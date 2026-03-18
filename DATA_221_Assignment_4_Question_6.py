from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras import layers, models

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()