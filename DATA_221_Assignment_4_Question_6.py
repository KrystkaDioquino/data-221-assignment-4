import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Load the fashion_mnist data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the images to include the channel dimension of (28, 28, 1)
X_train_reshaped = X_train.reshape((-1, 28, 28, 1))
X_test_reshaped = X_test.reshape((-1, 28, 28, 1))

# Build the CNN model
fashion_mnist_ccn_model = models.Sequential([

    # This learns spatial patterns like edges and shapes
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # This reduces the image size to highlight important features
    layers.MaxPooling2D((2, 2)),

    # Converts the 2D feature maps into a 1D vector for the output layer
    layers.Flatten(),

    # Dense output layer with 10 nodes. One for each fashion category
    layers.Dense(10, activation='softmax')])

# Compile the model
fashion_mnist_ccn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 15 epochs
fashion_mnist_ccn_model.fit(X_train_reshaped, y_train, epochs=15, verbose=1)

# Report the test accuracy
test_loss, test_accuracy = fashion_mnist_ccn_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"\nCNN Model Test Accuracy: {test_accuracy:.2f}")

"""
CNNs are much preferred for image data because using an MLP would flatten the image into a huge vector with hundreds of thousands of weights.
In CNN, it uses filters that slide across the entire image while only connecting to nearby pixels, efficiently learning simple patterns like edges and corners.

In Fashion MNIST, convolution layers learn basic edges and lines, while later layers combine these into complex textures and shapes of clothes. 
Filters automatically discover these clothing patterns without manual feature design.
"""

