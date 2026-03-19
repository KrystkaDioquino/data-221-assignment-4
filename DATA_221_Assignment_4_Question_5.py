from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Loads the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Assign the feature matrix and target vector
X = breast_cancer_data.data
y = breast_cancer_data.target

# Split to 80/20 train-test with random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Decision Tree
# Modify constraints with max_depth and min_samples_split
breast_cancer_constrained_tree = DecisionTreeClassifier(max_depth = 4, min_samples_split= 10, random_state= 42)

# Train the constrained tree model
breast_cancer_constrained_tree.fit(X_train, y_train)
breast_cancer_constrained_tree_predictions = breast_cancer_constrained_tree.predict(X_test)

# Compute the confusion matrix of the model
decision_tree_confusion_matrix = confusion_matrix(y_test, breast_cancer_constrained_tree_predictions)

# Display the confusion matrix
print(f"Confusion Matrix for Decision Tree: \n{decision_tree_confusion_matrix}")

#Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
breast_cancer_nn_model = MLPClassifier(hidden_layer_sizes=(10,), activation="logistic", max_iter=500, random_state=42)
breast_cancer_nn_model.fit(X_train_scaled, y_train)

# Predict testing data
breast_cancer_nn_model_test_predictions =  breast_cancer_nn_model.predict(X_test_scaled)

# Compute confusion matrix
neural_network_confusion_matrix = confusion_matrix(y_test, breast_cancer_nn_model_test_predictions)

# Display the confusion matrix
print(f"Confusion Matrix for Neural Network: \n{neural_network_confusion_matrix}")

"""
The Neural Network is the better choice for this task. In cancer screening, missing a positive case or a False Negative is the most dangerous 
mistake, and the Neural Network only missed one case while the Decision Tree missed three.

For Decision Tree, the advantage is that it does not require feature scaling and it will still work perfectly without extra preprocessing.
The limitation is it is sensitive to small changes in the data, where changing just a few rows of training data can result in a completely 
different tree structure.

For Neural Network, the advantage is that it can be easily improved by adding more layers or neurons, allowing it to handle much larger and 
more difficult datasets than a basic tree. The limitation is It requires much more data and time to train properly, as its complex math needs 
many examples to avoid making random guesses.
"""