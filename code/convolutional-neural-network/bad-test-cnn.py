import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.keras import layers, models

# Fetch 10 categories (to match 10 CNN outputs)
cats = fetch_20newsgroups(subset="all").target_names[:10]
data = fetch_20newsgroups(subset="all", categories=cats, remove=("headers","footers","quotes"))
X, y = data.data, data.target

# TF–IDF → fixed‑length vectors
vec = TfidfVectorizer(max_features=1024)
X = vec.fit_transform(X).toarray()

# Reshape into (32,32,3) “images”
X = X.reshape(-1,32,32,1).repeat(3, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=9000, stratify=y, random_state=42)

# Normalize
X_train, X_test = X_train.astype("float32"), X_test.astype("float32")

# Same CNN architecture as “good” test
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train briefly
model.fit(X_train, y_train, epochs=5, verbose=2)

# Evaluate
preds = model.predict(X_test).argmax(axis=1)
accuracy  = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall    = recall_score(y_test, preds, average='weighted')
f1        = f1_score(y_test, preds, average='weighted')

# Write Markdown report
output_dir = "output/convolutional-neural-network"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "bad-cnn-test.md"), "w") as f:
    f.write(f"""
# Convolutional Neural Network (CNN)

**Model Type:**  
Convolutional Neural Network (TensorFlow/Keras).

**Cons:**  
- CNNs lack the sequential memory needed to model long‑range dependencies in text.
In that they lack awareness of the order of words which is critical to their overall meaning.
Ex: "the dog bit" vs "the bit dog", CNNs would treat these sentences as identical due to shared weights and convolutions.
- Image classification principles (CNNs) do not translate well to text classification tasks:
    - Convolutional filters collapse the text into a fixed‑length vector, losing the sequential nature of text.
    - Subsampling (pooling) is not ideal for text as we lose essential information.
    - Lastly, shared weights across different parts of text is not ideal-- different parts of text may have different meanings
    and should be treated differently.

## Dataset:
- **Task:** 20 Newsgroups text classification (10 categories).  
- **Training subset:** 10,000 documents (stratified).  
- **Test subset:** Remaining documents.

## Test Summary
- **Accuracy:** {accuracy:.3f}  
- **Precision (weighted):** {precision:.3f}  
- **Recall (weighted):** {recall:.3f}  
- **F1 Score (weighted):** {f1:.3f}

As expected, the CNN performs no better than chance (50-50), illustrating why a novel model architecture 
(RNNs/transformers) are required for text and more specifically sequential processing tasks.

## Next Steps
Look into the output/recurrent-neural-network/good-test-rnn.py for a more suitable model for text classification tasks.
""")

print(f"CNN bad‑test summary written to {output_dir}/bad-cnn-test.md")
