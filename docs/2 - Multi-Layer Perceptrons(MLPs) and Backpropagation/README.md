## Brief:

Here we explore a key enhancement to the basic perceptron discussed in the last section-- the multi-layered perceptron(MLP) and the mathematical tool of backpropagation that helps systematically optimize MLPs.

## Paper 1: [Learning Representations by Backpropagating Errors](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)

- **Date Published:** 1986
- **Authors** - David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams

### Key Idea:

Given a simple multi-layered neural network with an input layer, several intermediate layers and an output layer you can mathematically calculate the gradient of the error with respect to the the weights and inputs and then utilize gradient descent to update weights in the direction that reduces error. As practically observed, the algorithm over time converges to local optima usually not much worse than than true global optima.

### Methodological Essentials:

- Firstly, we define an input layer, multiple intermediate layers, and an output layer.
- Input to each node: Linear summation on weighted outputs from previous nodes.
- Output of each node: A non-linear output on the overall input to each node.
- Total error function: 1/2 $\Sigma_{c}(\Sigma_{j} (y_{j,c} - d_{j, c})) $ where y corresponds to the prediction and d corresponds to the true output vector
- Via multivariable calculus we can find the gradient of the Error function with respect to each weight and propagate this backwards through the network.
- At the end we have an overall gradient for each input-output pair with respect to each weight and we can update each weight in the direction with reduces Error overall.
- We can repeat the above procedure until convergence.

### Valuable Insights/Conclusions:

1) **Mathematical Optimization(Calculus) For Machine Learning --> Underlies All Modern Machine Learning:** The use of gradients to optimize error functions underlies all of modern machine learning from supervised learning with regression/classification, and unsupervised learning with methods like PCA, and obviously deep learning with models like the multi-layered perceptron covered in this paper. In more modern models the full optimization of models isn't always feasible, however, the idea of mathematical modelling as a gateway to improving machine learning solutions is a key idea that permeates to all solutions even today.
2) **Non-Linear Output Function --> Non-Linear Activation Functions**: The author utilized non-linear output functions for each node instead of a simple linearly weighted sum of inputs and weights which I found very interesting. The implicit advantage here is the ability for the network to capture non-linear relations in data that traditional linear predictors may struggle to map to. Similar to how we may apply linear activation functions to output nodes in modern neural networks which is pretty much necessary otherwise we fail to capture non-linear relationships properly.
3) **Intermediate Layers Represent "Useful Features" --> Layered Representation of Knowedge:** The intermediate layers learn to transform raw input data into progressively more abstract representations, capturing the essential features needed for the final task. Early layers might identify simple patterns (like edges in an image or basic sound frequencies in audio), while deeper layers combine these to form higher-level concepts (such as shapes or objects). This idea is central to modern deep learning; it’s why architectures like CNNs, RNNs, and transformers are so effective. They automatically learn features that not only reduce the dimensionality of the data but also enhance the model’s ability to generalize. This process, known as representation learning, underpins many state-of-the-art models across computer vision, natural language processing, and beyond.
