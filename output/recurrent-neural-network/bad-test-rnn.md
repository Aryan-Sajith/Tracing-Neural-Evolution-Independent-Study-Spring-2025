
# Baseline RNN — Parity Classification

**Model Type:**  
Fast RNN with Bidirectional GRU (same baseline as good test with RNNs)

**Task:**  
Classify the parity (even vs odd sum) of randomly generated binary sequences of length 500.
Ex: 
[0, 1, 0, 1, 1] -> 1 (odd sum)
[1, 0, 1, 0, 0] -> 0 (even sum)  

**Cons:**  
- Parity is a global property requiring integration over the entire sequence. Without capturing the whole sequence, 
the model struggles on this task.
- Vanishing Gradients- RNNs have difficulty learning long-range dependencies due to vanishing gradients. In that, 
the gradients become so small that they do not affect the weights, and the model does not learn leading to stagnation in model performance. 

## Test Summary
- **Accuracy:** 0.483  
- **Precision (weighted):** 0.484  
- **Recall (weighted):** 0.483  
- **F1 Score (weighted):** 0.476

This baseline RNN performs close to random guessing (~50%), highlighting its inability to capture global sequence dependencies. 
Transformers—with self‑attention that directly relates every token to every other—are far better suited for tasks requiring 
such long‑range reasoning.

## Next Steps
Look at the output/transformer/good-test.md for a comparison of the same task using a transformer model.
