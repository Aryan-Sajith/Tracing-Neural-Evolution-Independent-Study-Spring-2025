import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the MLP classifier
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=20, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision (weighted): {precision:.3f}")
print(f"Recall (weighted): {recall:.3f}")
print(f"F1 Score (weighted): {f1:.3f}")

# Write results of good mlp test to a output markdown file
os.makedirs("output/multi-layer-perceptron", exist_ok=True)
md_content = f"""
# Multilayer Perceptron (MLP)

**Model Type:**  
Multilayer Perceptron classifier (scikit‑learn) using backpropagation.

**Pros:**  
- Capable of learning non‑linear decision boundaries.  
- Simple API and fast convergence via Adam optimizer.  
- Scales to high‑dimensional data (e.g., images).

# Positive Test Results for MLP:
## Dataset:
- MNIST dataset: 70,000 handwritten‑digit images (28×28 pixels).  
- Features: pixel intensity values (784 features).  
- Labels: digits 0–9.

## Test Summary
- **Accuracy:** {accuracy:.3f}  
- **Precision:** {precision:.3f}  
- **Recall:** {recall:.3f}  
- **F1 Score:** {f1:.3f}

As shown above, the MLP classifier achieves high performance on the MNIST classification task, demonstrating its ability to learn complex, non‑linear patterns and generalize effectively to unseen data.

# Next Steps
Look at output/multi-layer-perceptron/bad-mlp-test to understand the limitations of a baseline MLP model.
"""
with open("output/multi-layer-perceptron/good-mlp-test.md", "w") as f:
    f.write(md_content)

print("MLP test summary written to output/multi-layer-perceptron/good-mlp-test.md")