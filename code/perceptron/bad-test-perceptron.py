import os
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

# Tracks the output of bad test for perceptron
# Ensure the output directory exists
os.makedirs("output/perceptron", exist_ok=True)

# Create the markdown content for the good test
bad_md_content = f"""
# Perceptron

**Model Type:**  
Base perceptron: Utilizes linearly weighted inputs to solve binary classification problems.

**Cons:**  
- Does not perform well on non-linearly separable data.  
- Fails to learn complex patterns (e.g., XOR) without modifications.

# Negative Test Results for Perceptron:
## Dataset:
- Iris dataset: Contains 100 samples of Iris-virginica and Iris-versicolor flowers.
- Features: sepal length, sepal width, petal length, and petal width.
- Labels: -1 for Iris-virginica and 1 for Iris-versicolor.
- These classes are known to be non-linearly separable on these 4 features.

## Test Summary
- **Accuracy:** {accuracy:.3f}
- **Precision:** {precision:.3f}
- **Recall:** {recall:.3f}
- **F1 Score:** {f1_score:.3f}

As can be seen above, the perceptron model performs comparably poorly on non-linearly separable classes within the Iris dataset, 
achieving lower accuracy, recall and F1 while maintaining a perfect precision. This is expected since the two-selected classes
(Iris-virginica and Iris-versicolor) are known to be non-linearly separable, which is a not good fit for the perceptron model. 

# Further Exploration: What Comes Next?
The perceptron model is not suitable for non-linearly separable data. To handle such cases, non-linear enhancements to the 
model were invented. One such enhancement is the Multi-Layer Perceptron (MLP), which can learn complex patterns by introducing
hidden layers between inputs and outputs(modelled after the multi-layered nature of the human mind), non-linear activation functions,
and backpropagation which allows for mathematical optimization of model weights. This model is explored in the multi-layer-perceptron output folder.
"""

# Write the markdown content to file
with open("output/perceptron/bad-perceptron-test.md", "w") as f:
    f.write(bad_md_content)

print("Bad perceptron test summary written to output/perceptron/bad-perceptron-test.md")
