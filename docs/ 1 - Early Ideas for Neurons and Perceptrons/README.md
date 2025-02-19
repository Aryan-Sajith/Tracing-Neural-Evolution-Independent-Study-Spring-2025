## Brief:
Here we explore early ideas of mathematical models for neurons and the perceptron model that came about in the mid 1900s(1940 - 1960):

## Paper 1: [A Logical Calculus of the Ideas Immanent in Nervous Activity:](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf)
- **Date Published:** 1943
- **Authors** - Warren S. Mcculloch And Walter Pitts

### Key Idea:
Given the all-or-nothing(true/false) nature of neurons, their behavior and function within a larger neural network can be effectively modeled with propositional logic.

### Methodological Essentials:
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

## Paper 2: [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain:](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)
- **Date Published:** 1958
- **Authors** - Frank Rosenblatt

### Key Idea:
Given the non-deterministic and seemingly random anture of neurons and neural activity learning can be though of moreso as a collective, distributed and probabilistically driven phenomena as opposed to a deterministic propositional system. This led to the general, mathematical model of a neuron known as the perceptron.

### Methodological Essentials:
1) We start by departing from the boolean framework of paper 1 and utilizing probability to model how information may be stored in the brain. The primary goal is statistical separability of distinct stimuli as achieved by the below system:
2) A rudimentary framework that underlie modern NNs today is pioneered:
  - Sensory Inputs(S-points): These units respond in an all-or-nothing(activated or not) fashion to incoming stimuli. These most closely resemble the neuron model as presented in paper 1
  - Association Cells(A-points): Arranged in collections or groups where they receive weighted inputs from the Sensory layer. Their activation depends on the balance of excitary/inhibitory inputs, subject to a fixed threshold.
  - Response Layer(R-points): This layer integrates outputs from the association layer with similar excitary/inhibitory inputs balance the exclusivity of responses.
3) The rest of the paper analyzes, tests and compares statistically relevant measures like Pa(The probability of A-units activation by a given stimulus), Pc(The conditional probability that an A-unit fires for one stimulus will also fire for another), and so on. With these these measures analyses of learning curves and performance metrics(eg: probability of correct recall or generalization) as functions of physical parameters like number of excitary/inhibitory connections, thresholds and network structure are performed.
4) The perceptron learns via tweaking the weights of connections in the association cells based on reinforcement, and early form of network weight optimization. Two forms of reinforcement are considered:
  - Monovalent(Only positive reinforcement): Active cells gain value with each reinforcement
  - Bivanelt(Positive and negative reinforcement): Active cells can gain or lose value to perform trial-and-error learning
5) Interestingly, two phases of responses are considered:
  - Predominant: Where a subset of association cells respond to a stimulus
  - Postdominant: Where one set of response units respond to a stimulus whereby alternative association cells are ignored or not impactful.

### Valuable Insights/Conclusions
1) **Selective Memory & Low-Level Pattern Recognition --> Pre-Deep-Reasoning LLMs** An interesting observation was the perceptron's ability to learn specific associations from a large number of inputs. This likely occurs since the network is inclined to reinforce pathways that lead to positive outcomes and minimize pathways which do not. This means that the perceptron can learn multiple pathways, respond even if some damage occurs and perform reasonably well. This interestingly matches up with much of modern LLMs such as ChatGPT 4, Claude AI, Google Gemini and so on where models built strong selective memory on training data and recognized patterns at a remarkable scale with neural architectures. Similar to the perceptron, however, the lack of scale, architectural design and evolution prevented deeper reasoning and reflection in these models until very recently with models with ChatGPT O1, Deepseek and so on.
2) **Performance Parallel with Brain Damanged Patients --> Implies that novel architectural solutions are necessary to elevate language model past low-level token-completion engines:** When the perceptron was faced with tasks that required deeper processing and more nuanced neural activity exhibits behavior reminiscent of brain-damaged patients. In such situations the perceptron may recall surface-level details and concepts but struggled with relational reasoning and analysis reminiscent of healthy brains. This suggests that deeper architectures with more layers(led to several innovations in neural networks over the years), more layered reasoning(like Chain of Thought), and so on are needed for higher-order reasoning capabilities.
