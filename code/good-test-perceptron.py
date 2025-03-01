import numpy as np
from sklearn import datasets
from perceptron import Perceptron
from sklearn.model_selection import train_test_split

# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data[:100]  # Selecting first 100 samples (Iris-setosa and Iris-versicolor)
y = iris.target[:100]

# Convert target labels to binary: 0 (Iris-setosa) -> -1, 1 (Iris-versicolor) -> 1
y = np.where(y == 0, -1, 1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create, train and predict using the perceptron model
model = Perceptron(learning_rate=0.01, num_iters=10)
model.fit(X_train, y_train)
predictions = model.step_function_prediction(X_test)

# Evaluation Metrics:
# - Accuracy: The proportion of correctly classified instances
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")
# - Precision: The proportion of relevant retrieved instances among all retrieved instances
true_positives = np.sum((predictions == 1) & (y_test == 1))
false_positives = np.sum((predictions == 1) & (y_test == -1))
precision = true_positives / (true_positives + false_positives)
print(f"Precision: {precision}")
# - Recall: The proportion of relevant retrieved instances among all relevant instances
recall = true_positives / np.sum(y_test == 1)
print(f"Recall: {recall}")
# - F1 Score: The harmonic mean of precision and recall
f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1_score}")
