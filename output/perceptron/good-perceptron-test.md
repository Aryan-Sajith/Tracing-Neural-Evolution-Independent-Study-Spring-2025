
# Perceptron

**Model Type:**  
Base perceptron: Utilizes linearly weighted inputs to solve binary classification problems.

**Pros:**  
- Works well when dealing with linearly separable data.  
- Simple to implement and understand.  
- Fast training on simple datasets.

# Postive Test Results for Perceptron:
## Dataset:
- Iris dataset: Contains 100 samples of Iris-setosa and Iris-versicolor flowers.
- Features: sepal length, sepal width, petal length, and petal width.
- Labels: -1 for Iris-setosa and 1 for Iris-versicolor.
- These classes are known to be linearly separable on these 4 features.

## Test Summary
- **Accuracy:** 1.000
- **Precision:** 1.000
- **Recall:** 1.000
- **F1 Score:** 1.000

As can be seen above, the perceptron model performs well on linearly separable classes within the Iris dataset, 
achieving high accuracy, precision, recall, and F1 score. This is expected since the dataset is linearly separable, 
which is a good fit for the perceptron model. 

# What about non-linearly separable data?
Look at the output/perceptron/bad-perceptron-test file to see how the perceptron model performs on non-linearly separable data.
