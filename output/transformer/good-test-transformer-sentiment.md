
# Transformer (DistilBERT) — IMDB Sentiment Classification

**Model Type:**
Pre-trained Transformer (DistilBERT - `distilbert-base-uncased`) fine-tuned for sequence classification. Utilizes Hugging Face `transformers` and `datasets` libraries.

**Task:**
Classify movie reviews from the IMDB dataset as positive (1) or negative (0). This is a standard benchmark task for NLP.

**Pros (Why Transformer excels here):**
- **Contextual Understanding:** Self-attention allows the model to weigh the importance of different words in the review relative to each other, capturing context, negation, and nuances crucial for sentiment.
- **Pre-training:** Leverages knowledge learned from vast amounts of text data during pre-training, requiring less task-specific data and fewer epochs for good performance (fine-tuning).
- **Handles Variable Length:** Transformers, with appropriate tokenization (padding/truncation), effectively process sequences of varying lengths common in real-world text.

**Experiment Setup:**
- Dataset: `imdb` (subset: 2000 train, 500 test)
- Epochs: 1
- Batch Size: 16
- Optimizer: Adam (LR=5e-05)

## Test Summary
- **Accuracy:** 0.888
- **Precision (weighted):** 0.891
- **Recall (weighted):** 0.888
- **F1 Score (weighted):** 0.888

**Analysis:**
The fine-tuned DistilBERT model achieves high performance on the IMDB sentiment classification task, even with limited training data and epochs. This demonstrates the power of pre-trained Transformers for understanding language and performing classification tasks effectively. The self-attention mechanism is key to capturing the relationships between words that determine sentiment.

## Next Steps
Use a variational task for which transformers may struggle in the bad-test-transformer-sentiment.md file.
