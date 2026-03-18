from sklearn.datasets import load_breast_cancer
import pandas as pd

# Loads the breast cancer dataset
breast_cancer_data = load_breast_cancer()

# Assign the feature matrix and target vector
X = breast_cancer_data.data
y = breast_cancer_data.target

# Display the shape of the feature matrix and target vector
print("The shape of feature matrix X: ", X.shape)
print("The shape of the target vector y: ", y.shape)

# Count the total number of sample per classification
number_of_samples = pd.Series(y).value_counts().sort_index()

# Prints out the number of samples of class 0 and class 1
print("Samples per Class:")
print(f"{breast_cancer_data.target_names[0]}: {number_of_samples[0]}")
print(f"{breast_cancer_data.target_names[1]}: {number_of_samples[1]}")


"""
When converted to percentages, Malignant cases represent 37.3% of the data while Benign cases represent 62.7%. 
Since this distribution deviates significantly from a 50-50 split, the dataset is imbalanced. Maintaining balance ensures 
the model can effectively learn the underlying patterns for both classes, preventing it from developing a predictive bias toward the majority class.
"""