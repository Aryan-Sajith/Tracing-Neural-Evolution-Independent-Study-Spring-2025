import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

# Load the Iris dataset
iris = datasets.load_iris()
# Select samples for Iris-versicolor (label 1) and Iris-virginica (label 2)
X = iris.data[50:]  # Samples 50 to 149 (100 samples total)
y = iris.target[50:]
# Convert labels: assign 1 to Versicolor and -1 to Virginica
y = np.where(y == 1, 1, -1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the perceptron
model = Perceptron(learning_rate=0.01, num_iters=25)
model.fit(X_train, y_train)

# Get predictions on the test set
predictions = model.step_function_prediction(X_test)

# Calculate evaluation metrics
accuracy = np.mean(predictions == y_test)
true_positives = np.sum((predictions == 1) & (y_test == 1))
false_positives = np.sum((predictions == 1) & (y_test == -1))
false_negatives = np.sum((predictions == -1) & (y_test == 1))

precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

print(f"Test Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1_score:.3f}")
