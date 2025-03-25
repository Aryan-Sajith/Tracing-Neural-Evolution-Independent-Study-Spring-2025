import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Synthetic parity classification
# The task is to classify the parity (even vs odd sum) of binary sequences.
# Ex: 
# [0, 1, 0, 1, 1] -> 1 (odd sum)
# [1, 0, 1, 0, 0] -> 0 (even sum)
SEQ_LEN    = 500 # specify sequence length
VOCAB_SIZE = 2      # binary tokens {0,1}
N_SAMPLES  = 10_000

# Generate random binary sequences + parity labels (0=even sum, 1=odd sum)
X = np.random.randint(0, VOCAB_SIZE, size=(N_SAMPLES, SEQ_LEN))
y = np.sum(X, axis=1) % 2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=8_000, stratify=y, random_state=42
)

# Create TensorFlow datasets from raw NumPy arrays
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(8_000).batch(32).cache().prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(32).cache().prefetch(tf.data.AUTOTUNE)

# Baseline RNN (same architecture as good test)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 128, input_length=SEQ_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation="softmax"),
])

# Compile and train model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(train_ds, epochs=3, verbose=2)

# Evaluate model
preds = model.predict(test_ds).argmax(axis=1)
accuracy  = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")
recall    = recall_score(y_test, preds, average="weighted")
f1        = f1_score(y_test, preds, average="weighted")

# Write report
output_dir = "output/recurrent-neural-network"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "bad-test-rnn.md"), "w") as f:
    f.write(f"""
# Baseline RNN — Parity Classification

**Model Type:**  
Fast RNN with Bidirectional GRU (same baseline as good test with RNNs)

**Task:**  
Classify the parity (even vs odd sum) of randomly generated binary sequences of length {SEQ_LEN}.
Ex: 
[0, 1, 0, 1, 1] -> 1 (odd sum)
[1, 0, 1, 0, 0] -> 0 (even sum)  

**Cons:**  
- Parity is a global property requiring integration over the entire sequence. Without capturing the whole sequence, 
the model struggles on this task.
- Vanishing Gradients- RNNs have difficulty learning long-range dependencies due to vanishing gradients. In that, 
the gradients become so small that they do not affect the weights, and the model does not learn leading to stagnation in model performance. 

## Test Summary
- **Accuracy:** {accuracy:.3f}  
- **Precision (weighted):** {precision:.3f}  
- **Recall (weighted):** {recall:.3f}  
- **F1 Score (weighted):** {f1:.3f}

This baseline RNN performs close to random guessing (~50%), highlighting its inability to capture global sequence dependencies. 
Transformers—with self‑attention that directly relates every token to every other—are far better suited for tasks requiring 
such long‑range reasoning.

## Next Steps
Look at the output/transformer/good-test.md for a comparison of the same task using a transformer model.
""")

print(f"Bad RNN summary written to {output_dir}/bad-test-rnn.md")
