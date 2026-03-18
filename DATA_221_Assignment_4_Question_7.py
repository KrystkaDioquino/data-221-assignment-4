from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()