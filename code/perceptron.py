import numpy as np

class Perceptron:
    """Defines a foundational perceptron neural network model."""
    def __init__(self, learning_rate=0.01, num_iters=1_000):
        """
        Initializes the perceptron model with a passed in learning rate and number of iterations to
        train for.

        - learning_rate: A small float between 0 and 1 that determines the step size when updating weights(aka "learning")
        - num_iters: An integer that determines the number of iterations the model should train for
        """
        self.learning_rate = learning_rate
        self.num_iters = num_iters

    def weighted_sum(self, X):
        """Calculates the weighted sum of the input features and weights."""
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    def step_function_prediction(self, X):
        """Returns the predicted output of the input features via a step function."""
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        """
        Trains the perceptron model on the input features and target labels.
        """
        # Initialize weights(to 0s) and bias(to 1)
        self.weights = np.zeros(1 + X.shape[1])
        # Initialize errors list to track the number of misclassifications
        self.errors = []

        # Run the training loop for the number of iterations specified
        for _ in range(self.num_iters):
            error = 0

            print("Initial Bias & Weights:", self.weights)
            
            for x_i, true_y in zip(X, y):
                # First predict the output of input x_i
                prediction = self.step_function_prediction(x_i)

                # Then calculate the update to the weights
                update = self.learning_rate * (true_y - prediction)

                # Update the weights and bias
                self.weights[1:] += update * x_i # Since the weights correspond to inputs we scale by x_i
                print("Updated Weights:", self.weights[1:])
                self.weights[0] += update
                print("Updated Bias:", self.weights[0])

                # Update the error count if the prediction was incorrect
                error += 0 if update == 0 else 1
            
            # Append the error count for this training iteration
            self.errors.append(error)
        
        return self
    
