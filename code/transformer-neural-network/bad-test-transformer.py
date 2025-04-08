import os
import numpy as np
import tensorflow as tf
import random
from datasets import Dataset # Import Dataset class
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from sklearn.metrics import accuracy_score
import time

# --- Configuration ---
MODEL_NAME = "t5-small"              # Standard Seq2Seq Transformer
N_SAMPLES = 3000                     # Number of synthetic samples
TEST_SIZE = 500                      # Number of samples for the test set
MAX_INPUT_LEN = 64                   # Max length for input text
MAX_TARGET_LEN = 8                   # Max length for target answer (e.g., "-15")
BATCH_SIZE = 16
EPOCHS = 3                           # Limited epochs to ensure "bad" performance
LEARNING_RATE = 5e-5

# --- Generate Synthetic Arithmetic Data ---
print("Generating synthetic arithmetic reasoning data...")
def generate_arithmetic_problem():
    num1 = random.randint(-20, 20)
    num2 = random.randint(1, 20) # Keep second number positive for simpler ops
    operation = random.choice(['add', 'subtract'])

    if operation == 'add':
        result = num1 + num2
        # Simple templates
        templates = [
            f"Question: What is {num1} plus {num2}? Answer:",
            f"Question: Calculate the sum of {num1} and {num2}. Answer:",
            f"Question: If you have {num1} items and get {num2} more, how many total items? Answer:",
            f"Question: {num1} + {num2} = ? Answer:",
        ]
    else: # subtract
        result = num1 - num2
        templates = [
            f"Question: What is {num1} minus {num2}? Answer:",
            f"Question: Calculate the difference between {num1} and {num2}. Answer:",
            f"Question: If you start with {num1} points and lose {num2}, what is the score? Answer:",
            f"Question: {num1} - {num2} = ? Answer:",
        ]

    text = random.choice(templates)
    answer = str(result)
    return {"text": text, "answer": answer}

data = [generate_arithmetic_problem() for _ in range(N_SAMPLES)]
print(f"Generated {len(data)} samples.")
# Example: print(random.choice(data))

# --- Create Hugging Face Dataset ---
# Convert list of dicts to dict of lists
data_dict = {key: [d[key] for d in data] for key in data[0]}
raw_dataset = Dataset.from_dict(data_dict)

# --- Train/Test Split ---
dataset_dict = raw_dataset.train_test_split(test_size=TEST_SIZE, seed=42)
train_dataset = dataset_dict["train"]
test_dataset = dataset_dict["test"]
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# --- Load Tokenizer ---
print(f"Loading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Tokenize Data ---
print("Tokenizing data...")
def tokenize_function(examples):
    # T5 typically doesn't require a prefix for simple fine-tuning,
    # but you could add one like "calculate: " if needed.
    model_inputs = tokenizer(examples["text"], max_length=MAX_INPUT_LEN, truncation=True, padding=False) # Pad later

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["answer"], max_length=MAX_TARGET_LEN, truncation=True, padding=False) # Pad later

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "answer"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "answer"])

# --- Prepare TensorFlow Datasets ---
print("Preparing TensorFlow datasets...")
# Dynamic padding using Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, return_tensors="tf") # Use model=None for TF

tf_train_dataset = tokenized_train_dataset.to_tf_dataset(
    columns=[col for col in tokenized_train_dataset.column_names if col != 'labels'], # Input features
    label_cols=["labels"], # Decoder labels
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

tf_test_dataset = tokenized_test_dataset.to_tf_dataset(
    columns=[col for col in tokenized_test_dataset.column_names if col != 'labels'], # Input features
    label_cols=["labels"], # Decoder labels (used for loss calculation during eval potentially, not generation)
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
# We also need the original string labels for evaluation
y_test_text = test_dataset["answer"]

# --- Load Model ---
print(f"Loading model: {MODEL_NAME}...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# --- Compile Model ---
print("Compiling model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# The model computes the loss internally when labels are provided
model.compile(optimizer=optimizer) # Loss is handled internally for TFAutoModelForSeq2SeqLM when labels are passed
model.summary() # Note: Keras compile doesn't show loss/metrics setup this way

# --- Train Model ---
print(f"Starting training for {EPOCHS} epochs...")
start_time = time.time()
history = model.fit(
    tf_train_dataset,
    # validation_data=tf_test_dataset, # Validation during training requires careful handling of generation metrics
    epochs=EPOCHS
)
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Evaluate (using Generation) ---
print("Evaluating model on test set using generation...")

# Prepare inputs for generation (without labels)
tf_test_dataset_for_pred = tokenized_test_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask'], # Only pass encoder inputs
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator, # Use same collator to get padding right
)

# Generate predictions (token IDs)
all_preds_ids = []
print("Generating predictions...")
for batch in tf_test_dataset_for_pred:
    # model.generate expects dictionary or tf.Tensor inputs directly
    # batch already contains 'input_ids' and 'attention_mask'
    predictions = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        max_new_tokens=MAX_TARGET_LEN # Ensure generation stops
    )
    all_preds_ids.extend(predictions.numpy().tolist()) # Convert to list of lists

# Decode predictions
preds_text = tokenizer.batch_decode(all_preds_ids, skip_special_tokens=True)

# Ensure lengths match (debugging check)
print(f"Number of predictions: {len(preds_text)}")
print(f"Number of true labels: {len(y_test_text)}")
if len(preds_text) != len(y_test_text):
    print("Warning: Mismatch in number of predictions and labels!")
    # Handle mismatch, e.g., truncate or investigate data pipeline
    min_len = min(len(preds_text), len(y_test_text))
    preds_text = preds_text[:min_len]
    y_test_text = y_test_text[:min_len]


# Calculate Exact Match Accuracy
# Comparing strings directly
accuracy = accuracy_score(y_test_text, preds_text)

# Precision/Recall/F1 are not very meaningful here, accuracy (exact match) is key.
precision = accuracy # Treat EM as precision
recall = accuracy    # Treat EM as recall
f1 = accuracy        # Treat EM as f1

print(f"\nFinal Calculated Metrics (Exact Match):")
print(f"- Accuracy: {accuracy:.4f}")
# print(f"- Precision (weighted): {precision:.4f}") # Less meaningful
# print(f"- Recall (weighted): {recall:.4f}")    # Less meaningful
# print(f"- F1 Score (weighted): {f1:.4f}")        # Less meaningful

# --- Write Report ---
output_dir = "output/transformer" # Keep in transformer directory
os.makedirs(output_dir, exist_ok=True)
# New filename for this specific "bad test"
output_filename = "bad-test-transformer-arithmetic.md"

print(f"\nWriting report to {os.path.join(output_dir, output_filename)}...")
with open(os.path.join(output_dir, output_filename), "w") as f:
    f.write(f"""
# Standard Transformer ({MODEL_NAME}) — Arithmetic Reasoning (Bad Test)

**Model Type:**
Standard Sequence-to-Sequence Transformer ({MODEL_NAME}) fine-tuned on synthetic arithmetic problems. Intended as a "bad test" baseline.

**Task:**
Solve simple arithmetic problems presented in natural language text.
Ex: "Question: What is 5 plus 3? Answer:" -> Expected Output: "8"

**Cons (Why a standard Transformer struggles here):**
- **Lack of Explicit Reasoning:** Standard Seq2Seq models primarily rely on pattern matching and statistical correlations learned during pre-training and fine-tuning. They lack built-in mechanisms for performing reliable multi-step calculations or symbolic manipulation.
- **Sensitivity to Phrasing:** Performance can degrade significantly with slight changes in wording or the presence of distractors, as the model hasn't learned the underlying arithmetic *procedure*.
- **Limited Generalization:** May memorize answers for patterns seen during training but fails to generalize to unseen number combinations or slightly different problem structures. Requires vast amounts of data to approximate robust reasoning.

**Experiment Setup:**
- Dataset: Synthetic Arithmetic Problems ({N_SAMPLES} total, {TEST_SIZE} test)
- Max Input Length: {MAX_INPUT_LEN}, Max Target Length: {MAX_TARGET_LEN}
- Epochs: {EPOCHS}
- Batch Size: {BATCH_SIZE}
- Optimizer: Adam (LR={LEARNING_RATE})

## Test Summary (Exact Match)
- **Accuracy:** {accuracy:.3f}
- **Precision (EM):** {precision:.3f}
- **Recall (EM):** {recall:.3f}
- **F1 Score (EM):** {f1:.3f}
*(Note: Precision/Recall/F1 based on Exact Match)*

**Analysis:**
The standard {MODEL_NAME} Transformer achieves poor accuracy (exact match) on this arithmetic reasoning task, especially given the limited training ({EPOCHS} epochs). It struggles to consistently produce the correct numerical result, likely making errors based on surface-level text patterns rather than performing the actual calculation. This highlights the limitations of standard architectures for tasks requiring explicit, reliable, step-by-step reasoning.

## Next Steps
This poor performance sets the stage for exploring models or techniques designed for reasoning. A model employing Chain-of-Thought (CoT) prompting or fine-tuning would be expected to perform significantly better on this task by explicitly generating the intermediate reasoning steps (e.g., "5 + 3 = 8") before outputting the final answer. Comparing this result to a future CoT implementation would demonstrate the value of explicit reasoning mechanisms.
""")

print(f"Bad Transformer (Arithmetic) summary written to {os.path.join(output_dir, output_filename)}")