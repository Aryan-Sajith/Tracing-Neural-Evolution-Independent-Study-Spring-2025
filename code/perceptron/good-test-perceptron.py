import os
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

# Tracks the output of good test for perceptron
# Ensure the output directory exists
os.makedirs("output/perceptron", exist_ok=True)

# Create the markdown content for the good test
good_md_content = f"""
# Perceptron

**Model Type:**  
Base perceptron: Utilizes linearly weighted inputs to solve binary classification problems.

**Pros:**  
- Works well when dealing with linearly separable data.  
- Simple to implement and understand.  
- Fast training on simple datasets.

# Postive Test Results for Perceptron:
## Dataset:
- Iris dataset: Contains 100 samples of Iris-setosa and Iris-versicolor flowers.
- Features: sepal length, sepal width, petal length, and petal width.
- Labels: -1 for Iris-setosa and 1 for Iris-versicolor.
- These classes are known to be linearly separable on these 4 features.

## Test Summary
- **Accuracy:** {accuracy:.3f}
- **Precision:** {precision:.3f}
- **Recall:** {recall:.3f}
- **F1 Score:** {f1_score:.3f}

As can be seen above, the perceptron model performs well on linearly separable classes within the Iris dataset, 
achieving high accuracy, precision, recall, and F1 score. This is expected since the dataset is linearly separable, 
which is a good fit for the perceptron model. 

# What about non-linearly separable data?
Look at the [bad perceptron test](bad-test-perceptron.md) to see how the perceptron model performs on non-linearly separable data.
"""

# Write the markdown content to file
with open("output/perceptron/good-perceptron-test.md", "w") as f:
    f.write(good_md_content)

print("Good perceptron test summary written to output/perceptron/good-perceptron-test.md")
