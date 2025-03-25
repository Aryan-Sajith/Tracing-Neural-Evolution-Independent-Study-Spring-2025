
# Perceptron

**Model Type:**  
Base perceptron: Utilizes linearly weighted inputs to solve binary classification problems.

**Cons:**  
- Does not perform well on non-linearly separable data.  
- Fails to learn complex patterns (e.g., XOR) without modifications.

# Negative Test Results for Perceptron:
## Dataset:
- Iris dataset: Contains 100 samples of Iris-virginica and Iris-versicolor flowers.
- Features: sepal length, sepal width, petal length, and petal width.
- Labels: -1 for Iris-virginica and 1 for Iris-versicolor.
- These classes are known to be non-linearly separable on these 4 features.

## Test Summary
- **Accuracy:** 0.850
- **Precision:** 1.000
- **Recall:** 0.750
- **F1 Score:** 0.857

As can be seen above, the perceptron model performs comparably poorly on non-linearly separable classes within the Iris dataset, 
achieving lower accuracy, recall and F1 while maintaining a perfect precision. This is expected since the two-selected classes
(Iris-virginica and Iris-versicolor) are known to be non-linearly separable, which is a not good fit for the perceptron model. 

# Further Exploration: What Comes Next?
The perceptron model is not suitable for non-linearly separable data. To handle such cases, non-linear enhancements to the 
model were invented. One such enhancement is the Multi-Layer Perceptron (MLP), which can learn complex patterns by introducing
hidden layers between inputs and outputs(modelled after the multi-layered nature of the human mind), non-linear activation functions,
and backpropagation which allows for mathematical optimization of model weights. This model is explored in the multi-layer-perceptron output folder.
