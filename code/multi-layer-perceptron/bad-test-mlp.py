import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()

# Flatten images
X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)

# Use only a subset for speed (optional)
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=10000, random_state=42, stratify=y_train)
X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=2000, random_state=42, stratify=y_test)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLPClassifier (same architecture as “good” test)
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy  = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall    = recall_score(y_test, predictions, average='weighted')
f1        = f1_score(y_test, predictions, average='weighted')

# Write results to markdown
os.makedirs("output/multi-layer-perceptron", exist_ok=True)
md = f"""
# Multilayer Perceptron (MLP) — Poor Performance on CIFAR‑10

**Model Type:**  
Multilayer Perceptron classifier (scikit‑learn).

**Task:**  
CIFAR‑10 image classification (10 classes; 32×32 color images).

**Cons of MLP:**  
MLPs struggle with certain complex image‑classification tasks due to the following reasons:
- **Loss of Spatial Information:** MLPs do not inherently understand spatial relationships in images. 
Meaning that they do not understand that pixels close to each other are more related than pixels far apart,
this lack of context makes it difficult for MLPs to learn spatial features like edges, textures, and shapes.
This is a critical limitation for more complex image classification tasks beyond MNIST.
- **Parameteric Efficiency:** MLPs require a large number of parameters to learn spatial features.
This is because each neuron in the first hidden layer must learn a separate weight for each pixel in the input image whereas
you can share weights across abstracted image regions to more efficiently learn spatial features. For a simple example,
instead of learning a separate weight for each pixel in a 28x28 image, you can learn a single weight for each 2x2 region.
- **Lack of Hierarchical Feature Learning:** MLPs do not learn hierarchical features.
In image classification, features are often hierarchical, where lower layers learn basic features like edges and textures,
and higher layers learn more complex features like shapes and objects. Simple MLPs do not have this hierarchical feature learning capability 
since they treat all input features as structurally equivalent. Instead of this, you can learn hierarchical features by 
using abstracted representations of image features in increasing order of complexity. For a simple example, you can use a pooling layer to 
abstractly represent edge features learned in the first hidden layer, and then use this abstracted representation as input to the second hidden layer.
Such hierarchical feature learning allows you to learn more complex features with fewer parameters and less data. This is precisely what 
convolutional neural networks (CNNs) do.

## Test Summary & Results for MLP:
- **Accuracy:** {accuracy:.3f}  
- **Precision (weighted):** {precision:.3f}  
- **Recall (weighted):** {recall:.3f}  
- **F1 Score (weighted):** {f1:.3f}

As can be seen from the results, the MLP struggles to map the CIFAR‑10 images to their correct classes.
This demonstrates that a baseline fully‑connected MLP struggles on natural‑image classification tasks that require spatial understanding 
and hierarchical feature learning. To address these limitations, we can use convolutional neural networks (CNNs) which are specifically
designed to learn spatial features and hierarchical representations in images. To see the benefits of CNNs, refer to the output/cnn/good-cnn-test.md file.
"""
with open("output/multi-layer-perceptron/bad-mlp-test.md", "w") as f:
    f.write(md)

print("Bad MLP test summary written to output/multi-layer-perceptron/bad-mlp-test.md")
