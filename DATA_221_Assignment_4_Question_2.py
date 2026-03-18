from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Loads the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Assign the feature matrix and target vector
X = breast_cancer_data.data
y = breast_cancer_data.target

# Split to 80/20 train-test with stratification and random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Train the Decision Tree classifier using entropy
breast_cancer_decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
breast_cancer_decision_tree.fit(X_train, y_train)

# Predict using the training set and testing set
breast_cancer_decision_tree_train_prediction = breast_cancer_decision_tree.predict(X_train)
breast_cancer_decision_tree_test_prediction = breast_cancer_decision_tree.predict(X_test)

# Compute the training and testing accuracy
breast_cancer_decision_tree_train_accuracy = accuracy_score(y_train, breast_cancer_decision_tree_train_prediction)
breast_cancer_decision_tree_test_accuracy= accuracy_score(y_test, breast_cancer_decision_tree_test_prediction)

# Display the computed accuracy for both prediction
print("Training accuracy: ", breast_cancer_decision_tree_train_accuracy)
print("Testing accuracy: ", breast_cancer_decision_tree_test_accuracy)