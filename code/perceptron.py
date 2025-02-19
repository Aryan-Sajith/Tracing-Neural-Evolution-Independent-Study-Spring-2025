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