
# Improved RNN — Bidirectional GRU

**Model Type:**  
Fast RNN with Bidirectional GRU

**Pros:**  
- Captures long‑range contextual dependencies over the entire document (MAX_LEN=500).  
- Bidirectional GRU learns from both directions.
    - **Note:** LSTM is far more computationally expensive and does not offer significant gains in this task; thus,
            we have opted for GRU which still significantly outperforms the CNN baseline.
- GlobalMaxPooling1D captures the most salient features from the entire sequence -- similar to CNN 
however with a more nuanced understanding of the order of words.
- Dropout layers prevent overfitting to the training data by randomly 
setting a small fraction of input units to 0 at each update during training time.

## Test Summary
- **Accuracy:** 0.769  
- **Precision (weighted):** 0.781  
- **Recall (weighted):** 0.769  
- **F1 Score (weighted):** 0.770

As can be seen above the model drastically outperforms CNNs by ~30%, demonstrating 
how RNNs are better suited for capturing long‑range sequential dependencies in text data.

## Next Steps
Look at output/recurrent-neural-network/bad-test-rnn.md for a look into what challenges RNNs face and how that led to
newer architectures like Transformers.
