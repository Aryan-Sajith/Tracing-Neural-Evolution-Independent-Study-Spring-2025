
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
- **Accuracy:** 0.476  
- **Precision (weighted):** 0.483  
- **Recall (weighted):** 0.476  
- **F1 Score (weighted):** 0.469

As expected, the CNN performs no better than chance (50-50), illustrating why a novel model architecture 
(RNNs/transformers) are required for text and more specifically sequential processing tasks.

## Next Steps
Look into the output/recurrent-neural-network/good-test-rnn.py for a more suitable model for text classification tasks.
