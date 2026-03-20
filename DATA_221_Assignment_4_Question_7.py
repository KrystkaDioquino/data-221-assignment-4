from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

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

# Generate predictions for the testing data
test_predictions = fashion_mnist_ccn_model.predict(X_test_reshaped)

# Convert predicted probabilities to class labels
test_predictions_labels = tf.argmax(test_predictions, axis=1)

# Create confusion matrix for the model
cnn_model_confusion_matrix = confusion_matrix(y_test, test_predictions_labels)
print(cnn_model_confusion_matrix)

# Define the class names
fashion_mnist_class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Convert to numpy array to get indices
cnn_prediction_list = test_predictions_labels.numpy()

# This looks for the indices where prediction mistakes occur
wrong_prediction_indices = []
for i in range(len(y_test)):
    if cnn_prediction_list[i] != y_test[i]:
        wrong_prediction_indices.append(i)
    if len(wrong_prediction_indices) == 3:
        break

# Display the first 3 mistakes with the actual and predicted label
print("\nFirst 3 Misclassified Images :")
print(f"1: Actual={fashion_mnist_class_names[y_test[wrong_prediction_indices[0]]]}, "
      f"Predicted={fashion_mnist_class_names[test_predictions_labels[wrong_prediction_indices[0]]]}")
print(f"2: Actual={fashion_mnist_class_names[y_test[wrong_prediction_indices[1]]]}, "
      f"Predicted={fashion_mnist_class_names[test_predictions_labels[wrong_prediction_indices[1]]]}")
print(f"3: Actual={fashion_mnist_class_names[y_test[wrong_prediction_indices[2]]]}, "
      f"Predicted={fashion_mnist_class_names[test_predictions_labels[wrong_prediction_indices[2]]]}")

"""
The CNN frequently confuses similar-looking clothes, such as shirts with coats and ankle boots with sandals, due to 
their comparable characteristics. One way to improve the model's prediction is to add a second convolutional layer before 
the final Dense layer to learn finer clothing details and stabilize training.
"""