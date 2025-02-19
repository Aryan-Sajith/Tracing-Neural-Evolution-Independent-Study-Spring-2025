## Brief
Here we explore early ideas of mathematical models for neurons and the perceptron model that came about in the mid 1900s(1940 - 1960):

## Paper 1: [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)
- **Date Published:** 1943
- **Authors** - Warren S. Mcculloch And Walter Pitts

### Key Idea
Given the all-or-nothing(true/false) nature of neurons, their behavior and function within a larger neural network can be effectively modeled with propositional logic.

### Methodological Essentials
1) Neurons either fire or they don't which can be modelled by proposional true or false respectively.
2) Neural networks, then, can be logically constructed as propositional statements combined via conjunctions, disjunctions, and negation of neural activation.
  - _Example:_ If a neuron requires n other neurons to activate, this can be represented by a conjunction of propositional activations
3) A key contribution of the paper is Temporal Propositional Expressions(TPEs) which integrates not only logical propositions but also allows for the logical modelling of neural activity over time.
4) The rest primarily focuses on generalizing TPEs, propositions and logical connectors to general classes of neural networks.

### Valuable Insights/Conclusions
1) **Collective vs Singular Activations --> Regularization/Dropout**: The paper notes that, at the time in 1943, no neuron had been observed firing from a singular synapse. Instead it takes excitation from several neighbouring neurons in a quick succession for a neuron to activate. This suggests that our mind, and by extension, models build on it may de-emphasize heavily weighting singular connections between neurons. This principle manifests in modern techniques like regularization and dropout which focus on preventing model overfitting by developing more distributed representations of knowledge as opposed to narrowly focused singular connections.
2) **Logical Irreversability --> Challenges in Explainability**: We see that the logical framework can effectively forward-propogate logical connections however information is lost and we cannot fully reconstruct inputs given an exact output state and the model that led to it. This suggests that our mind, and by extension, models built on it may need to rely on 1-way mathematical functions that pose challenges for modern research in areas like explainability of neural networks and their black-box nature.
  - _Example:_ For an interesting analogy, consider that given the final state of the mind when someone experiences depression, anxiety, and so on doesn't allow us to reconstruct or re-trace the exact inputs that led to their final state. Even if we could, given the non-deterministic behavior of the mind we may not run into the same outcomes again.
3) **Mathematical Modeling of the Mind --> All of Modern ML**: The rigorous and mathematical framework provided suggests that our mind, and by extension, models built on it could be effectively modeled via mathematical frameworks and systems which revolutionzed machine learning laid the foundation for the field as we know it today in practically any type of learning(supervised, unsupervised, etc.)


