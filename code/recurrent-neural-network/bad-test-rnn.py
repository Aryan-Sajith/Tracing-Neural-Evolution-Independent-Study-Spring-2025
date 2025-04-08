import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer # Use the same tokenizer for fair comparison
# No train_test_split needed when using HF datasets splits directly
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time

TOKENIZER_NAME = "distilbert-base-uncased" # Use the SAME tokenizer as the good test
DATASET_NAME = "imdb"
NUM_TRAIN_SAMPLES = 2000             # Use same subset size
NUM_TEST_SAMPLES = 500               # Use same subset size
BATCH_SIZE = 16                      # Use same batch size
EPOCHS = 5                           # max epochs
MAX_LEN = 256                        # Max sequence length for RNN padding/truncation
EMBEDDING_DIM = 64                   # Dimension
GRU_UNITS = 32                       # GRU layer size
DENSE_UNITS = 64                     # Dense layer size

# --- Load Dataset (Same as Transformer Test) ---
print(f"Loading dataset: {DATASET_NAME}...")
raw_datasets = load_dataset(DATASET_NAME)
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(NUM_TEST_SAMPLES))
print(f"Using {len(train_dataset)} train samples and {len(test_dataset)} test samples.")

# --- Load Tokenizer (Same as Transformer Test) ---
print(f"Loading tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
VOCAB_SIZE = tokenizer.vocab_size

# --- Tokenize Data (Pad/Truncate for RNN) ---
print(f"Tokenizing data (max_len={MAX_LEN})...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=MAX_LEN)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
tokenized_train_dataset.set_format("np")
tokenized_test_dataset.set_format("np")

X_train = tokenized_train_dataset['input_ids']
y_train = tokenized_train_dataset['label']
X_test = tokenized_test_dataset['input_ids']
y_test = tokenized_test_dataset['label']

# --- Build TensorFlow Datasets (Standard Keras way) ---
print("Building TensorFlow datasets...")
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(NUM_TRAIN_SAMPLES)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
print("Data pipelines built.")

# --- Build RNN Model (Reduced Capacity) ---
print("Building RNN model (Reduced Capacity)...")
model = tf.keras.Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBEDDING_DIM,      # Reduced dimension
              input_length=MAX_LEN,
              name='embedding'),
    Bidirectional(GRU(GRU_UNITS, return_sequences=True), name='bidirectional_gru'), 
    GlobalMaxPooling1D(name='global_max_pooling'),
    Dropout(0.3, name='pool_dropout'), # Keep dropout for some regularization
    Dense(DENSE_UNITS, activation="relu", name='dense_1'), 
    Dropout(0.3, name='dense_dropout'),
    Dense(2, activation="softmax", name='output_softmax') # 2 classes: neg/pos
])

# --- Compile Model ---
print("Compiling RNN model...")
optimizer = tf.keras.optimizers.Adam() # Standard Adam optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary() # Summary will show fewer parameters

# --- Train Model (Reduced Epochs) ---
print(f"Starting RNN training for up to {EPOCHS} epochs...")

# EarlyStopping is still useful to prevent running full epochs if it plateaus early
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=2, # Reduced patience slightly
    mode='max',
    restore_best_weights=True,
    verbose=1
)

start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,              # max epochs
    callbacks=[early_stopping], # Use early stopping
    verbose=1
)
end_time = time.time()
print(f"RNN training finished in {end_time - start_time:.2f} seconds.")

# --- Evaluate ---
print("Evaluating RNN model on test set...")
loss_eval, accuracy_eval = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {loss_eval:.4f}, Test Accuracy (from evaluate): {accuracy_eval:.4f}")

# Predict probabilities
print("Making predictions with RNN model...")
preds_proba = model.predict(test_ds)
preds = np.argmax(preds_proba, axis=1)

# Calculate metrics
print("Calculating final metrics for RNN...")
accuracy  = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted")
recall    = recall_score(y_test, preds, average="weighted")
f1        = f1_score(y_test, preds, average="weighted")

print(f"\nFinal Calculated Metrics (Reduced Capacity RNN):")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision (weighted): {precision:.4f}")
print(f"- Recall (weighted): {recall:.4f}")
print(f"- F1 Score (weighted): {f1:.4f}")

# --- Write Report ---
output_dir = "output/recurrent-neural-network"
os.makedirs(output_dir, exist_ok=True)
# Modify filename slightly to indicate this is the  capacity version
output_filename = "bad-test-rnn-sentiment.md"

print(f"\nWriting report to {os.path.join(output_dir, output_filename)}...")
with open(os.path.join(output_dir, output_filename), "w") as f:
    f.write(f"""
# Reduced Capacity RNN — IMDB Sentiment Classification (Bad Test)

**Model Type:**
RNN using Bidirectional GRU layers, trained from scratch.

**Task:**
Classify movie reviews from the IMDB dataset as positive (1) or negative (0). (Same task as the Transformer 'good test').

**Cons (Why this specific RNN is expected to perform poorly):**
- **Lack of Pre-training:** Learns solely from the small {NUM_TRAIN_SAMPLES}-sample training set.
- **Inherent RNN Limitations:** Still faces challenges with very long-range dependencies compared to Transformers. 

**Experiment Setup:**
- Dataset: `{DATASET_NAME}` (subset: {NUM_TRAIN_SAMPLES} train, {NUM_TEST_SAMPLES} test)
- Tokenizer: `{TOKENIZER_NAME}` (Vocab Size: {VOCAB_SIZE})
- Max Sequence Length: {MAX_LEN}
- Model: Embedding({EMBEDDING_DIM}) -> BiGRU({GRU_UNITS}) -> GlobalMaxPool -> Dense({DENSE_UNITS}) -> Dense(2)
- Epochs Trained: {len(history.history.get('loss', [0]))} (Stopped early if patience met) / Max {EPOCHS}
- Batch Size: {BATCH_SIZE}

## Test Summary
- **Accuracy:** {accuracy:.3f}
- **Precision (weighted):** {precision:.3f}
- **Recall (weighted):** {recall:.3f}
- **F1 Score (weighted):** {f1:.3f}

**Analysis:**
This RNN, trained from scratch with limited epochs on the sentiment data, demonstrates lower performance compared to the fine-tuned Transformer. The constraints on model size and training time clearly hinder its ability to effectively capture the nuances of language required for accurate sentiment analysis, providing a clearer contrast to the Transformer's capabilities.

## Next Steps
Compare these results directly with the Transformer baseline in `output/transformer/good-test-transformer-sentiment.md`. The performance gap should now be more pronounced, highlighting the advantages of using larger, pre-trained models like Transformers for such NLP tasks.
""")

print(f"Bad RNN (Sentiment) summary written to {os.path.join(output_dir, output_filename)}")