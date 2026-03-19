from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Loads the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Assign the feature matrix and target vector
X = breast_cancer_data.data
y = breast_cancer_data.target

# Split to 80/20 train-test with random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the feature matrix
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
breast_cancer_nn_model = MLPClassifier(hidden_layer_sizes=(10,), activation="logistic", max_iter=500, random_state=42)
breast_cancer_nn_model.fit(X_train_scaled, y_train)

# Predict using train and testing data
breast_cancer_nn_model_train_predictions = breast_cancer_nn_model.predict(X_train_scaled)
breast_cancer_nn_model_test_predictions =  breast_cancer_nn_model.predict(X_test_scaled)

# Compute accuracies
train_accuracy = accuracy_score(y_train,breast_cancer_nn_model_train_predictions)
test_accuracy = accuracy_score(y_test, breast_cancer_nn_model_test_predictions)

# Display the accuracy for training data and testing data
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

"""
Feature scaling is necessary because neural networks use a mathematical process called Gradient Descent to 
update their weights. By putting all features into a similar range, you create a symmetrical surface that allows 
the model to find the best settings much faster. 

An epoch represents one complete pass of the entire training dataset through the network, where every row of data 
is seen once. During this process, the model calculates its error and updates its internal weights to improve its 
accuracy for the next round.
"""