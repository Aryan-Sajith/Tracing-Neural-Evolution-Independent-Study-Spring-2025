
# Convolutional Neural Network (CNN)

**Model Type:**  
Convolutional Neural Network (TensorFlow/Keras).

**Pros:**  
- Leverages spatial hierarchies via convolutional filters.  
- Robust to translations, scale variations, and noise.  
- Efficient parameter sharing reduces overfitting.

## Dataset:
- **CIFAR‑10:** 60,000 32×32 color images across 10 classes.  
- **Training subset:** 10,000 images (stratified).  
- **Test subset:** 2,000 images (stratified).

## Test Summary
- **Accuracy:** 0.592  
- **Precision (weighted):** 0.610  
- **Recall (weighted):** 0.592  
- **F1 Score (weighted):** 0.590

As shown above, the CNN performs significantly better than the baseline MLP by around 20% across all metrics demonstrating
the power of convolutional neural networks in efficiently extracting spatial hierarchies from images.

## Next Steps
Look at output/convolutional-neural-network/bad-cnn-test.md for a bad CNN test and what novel architectures 
or techniques could be used to improve performance.
