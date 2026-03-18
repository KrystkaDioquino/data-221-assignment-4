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
