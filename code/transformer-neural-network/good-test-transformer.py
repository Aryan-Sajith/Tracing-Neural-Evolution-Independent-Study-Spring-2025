import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased" # A smaller, faster Transformer
DATASET_NAME = "imdb"                # Standard sentiment analysis dataset
NUM_TRAIN_SAMPLES = 2000             # Use a subset for faster demo
NUM_TEST_SAMPLES = 500               # Use a subset for faster demo
BATCH_SIZE = 16
EPOCHS = 1 # Fine-tuning often requires few epochs
LEARNING_RATE = 5e-5 # Common learning rate for fine-tuning transformers

# --- Load Dataset ---
print(f"Loading dataset: {DATASET_NAME}...")
# Load imdb dataset, splits are 'train', 'test', 'unsupervised' (we ignore unsupervised)
raw_datasets = load_dataset(DATASET_NAME)
# Select smaller subsets for faster training/evaluation
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(NUM_TEST_SAMPLES))
print(f"Using {len(train_dataset)} train samples and {len(test_dataset)} test samples.")

# --- Load Tokenizer ---
print(f"Loading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Tokenize Data ---
print("Tokenizing data...")
def tokenize_function(examples):
    # Truncate sequences longer than the model's max input size
    return tokenizer(examples["text"], truncation=True, padding=False) # Pad later with data collator

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove original text column, rename label column for TF compatibility
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
# 'label' is usually fine, but ensure it matches model expectations if needed

# --- Prepare TensorFlow Datasets ---
print("Preparing TensorFlow datasets...")
# Dynamic padding using Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_train_dataset.to_tf_dataset(
    columns=[col for col in tokenized_train_dataset.column_names if col != 'label'], # Features
    label_cols=["label"], # Labels
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

tf_test_dataset = tokenized_test_dataset.to_tf_dataset(
    columns=[col for col in tokenized_test_dataset.column_names if col != 'label'], # Features
    label_cols=["label"], # Labels
    shuffle=False, # No need to shuffle test data
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# --- Load Model ---
print(f"Loading model: {MODEL_NAME}...")
# num_labels=2 for binary sentiment (positive/negative)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- Compile Model ---
print("Compiling model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # Use from_logits=True as HF models output logits
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

# --- Train Model ---
print(f"Starting training for {EPOCHS} epochs...")
start_time = time.time()
history = model.fit(
    tf_train_dataset,
    validation_data=tf_test_dataset,
    epochs=EPOCHS
)
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Evaluate ---
print("Evaluating model on test set...")
# Get predictions (logits)
all_logits = model.predict(tf_test_dataset).logits
preds = np.argmax(all_logits, axis=1)

# Get true labels
y_test = np.concatenate([labels.numpy() for _, labels in tf_test_dataset], axis=0)

# Calculate metrics
print("Calculating final metrics...")
accuracy  = accuracy_score(y_test, preds)
# Use 'binary' average for precision/recall/f1 if only 2 classes, or 'weighted'/'macro' if more
# Since IMDB is binary (0 or 1), 'binary' with pos_label=1 (positive class) is common,
# but 'weighted' works generally and handles potential imbalance.
precision = precision_score(y_test, preds, average="weighted")
recall    = recall_score(y_test, preds, average="weighted")
f1        = f1_score(y_test, preds, average="weighted")

print(f"\nFinal Calculated Metrics:")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- Precision (weighted): {precision:.4f}")
print(f"- Recall (weighted): {recall:.4f}")
print(f"- F1 Score (weighted): {f1:.4f}")

# --- Write Report ---
output_dir = "output/transformer"
os.makedirs(output_dir, exist_ok=True)
output_filename = "good-test-transformer-sentiment.md" # New filename

print(f"\nWriting report to {os.path.join(output_dir, output_filename)}...")
with open(os.path.join(output_dir, output_filename), "w") as f:
    f.write(f"""
# Transformer (DistilBERT) — IMDB Sentiment Classification

**Model Type:**
Pre-trained Transformer (DistilBERT - `{MODEL_NAME}`) fine-tuned for sequence classification. Utilizes Hugging Face `transformers` and `datasets` libraries.

**Task:**
Classify movie reviews from the IMDB dataset as positive (1) or negative (0). This is a standard benchmark task for NLP.

**Pros (Why Transformer excels here):**
- **Contextual Understanding:** Self-attention allows the model to weigh the importance of different words in the review relative to each other, capturing context, negation, and nuances crucial for sentiment.
- **Pre-training:** Leverages knowledge learned from vast amounts of text data during pre-training, requiring less task-specific data and fewer epochs for good performance (fine-tuning).
- **Handles Variable Length:** Transformers, with appropriate tokenization (padding/truncation), effectively process sequences of varying lengths common in real-world text.

**Experiment Setup:**
- Dataset: `{DATASET_NAME}` (subset: {NUM_TRAIN_SAMPLES} train, {NUM_TEST_SAMPLES} test)
- Epochs: {EPOCHS}
- Batch Size: {BATCH_SIZE}
- Optimizer: Adam (LR={LEARNING_RATE})

## Test Summary
- **Accuracy:** {accuracy:.3f}
- **Precision (weighted):** {precision:.3f}
- **Recall (weighted):** {recall:.3f}
- **F1 Score (weighted):** {f1:.3f}

**Analysis:**
The fine-tuned DistilBERT model achieves high performance on the IMDB sentiment classification task, even with limited training data and epochs. This demonstrates the power of pre-trained Transformers for understanding language and performing classification tasks effectively. The self-attention mechanism is key to capturing the relationships between words that determine sentiment.

## Next Steps
Use a variational task for which transformers may struggle in the bad-test-transformer-sentiment.md file.`
""")

print(f"Good Transformer (Sentiment) summary written to {os.path.join(output_dir, output_filename)}")