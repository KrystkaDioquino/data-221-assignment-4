from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Loads the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Assign the feature matrix and target vector
X = breast_cancer_data.data
y = breast_cancer_data.target

# Split to 80/20 train-test with random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Modify constraints with max_depth and min_samples_split
breast_cancer_constrained_tree = DecisionTreeClassifier(max_depth = 4, min_samples_split= 10, random_state= 42)

# Train the constrained tree model
breast_cancer_constrained_tree.fit(X_train, y_train)
breast_cancer_constrained_tree_predictions = breast_cancer_constrained_tree.predict(X_test)

# Compute the accuracy of the model
test_accuracy = accuracy_score(y_test, breast_cancer_constrained_tree_predictions)

# Display the accuracy
print(f"Test Accuracy: {test_accuracy:.2f}")

# Get the importance value of each feature and the feature name
importance_value = breast_cancer_constrained_tree.feature_importances_
feature_names = breast_cancer_data.feature_names

# Get only the top 5 feature name based on their value
top_features = sorted(zip(importance_value, feature_names), reverse=True)[:5]

print("\nTop 5 Most Important Features:")

# Print out the top 5 features
for i, (importance, name) in enumerate(top_features, 1):
    print(f"{i}. {name}")

"""
Controlling complexity, like setting a max_depth, prevents a tree from growing more and memorizing 
every detail in the training data. This forces the model to ignore random noise and focus on broad patterns, 
which significantly improves its ability to make accurate predictions on new data.

Feature importance identifies which variables are the best at cleaning up the entropy in the data. By ranking these 
variables, it makes the decision tree interpretable. With this, one can see exactly which features the model relied 
on most to make its final choice.
"""