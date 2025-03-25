## Brief:

Here we explore a key enhancement to the multi-layer perceptron discussed in the last section-- a convolutional neural network designed specifically to capture essential spatial hierarchies of information within images.

# Paper 1: [Gradient-Based Learning Applied to Document Recognition](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf)

- **Date Published:** 1998
- **Authors** - Yann Lecunn, Leon Bottou, Yoshua Bengio, Patrick Haffner

### Key Idea:

Baseline MLPs struggle with capturing spatial hierarchies of visual information(increasingly complex representations that feed into each other like colors/edges -> shapes/forms -> people/animals) due to them treating all pixels as essentially structurally independent and lacking proper scaling when dealing with even relatively large images(eg: 40 x 40 pixel images and 100 hidden nodes = 40 x 40 x 100 = 160,000 weights!) due to learning weights for each pixel. Convolutional neural networks tackle this by using local receptive fields(compressional techniques that capture context over chunks of images as opposed to dealing with all pixels individually), replicated(shared) weights across increasing layers of abstraction, and subsampling(the representation of knowledge in an increasingly abstracted fashion where lower levels capture lower-order features and use that as a base to capture higher order features in subsequent layers).

### Methodological Essentials:
#### Key Insight
Rather than treating every pixel independently (as in a fully connected MLP), LeNet‑5 exploits the spatial structure of images by learning local, shift‑invariant features in a hierarchical fashion. Local means each hidden unit only “looks at” a small neighborhood of pixels (e.g. a 5×5 patch) rather than the entire image. This lets the network learn elementary visual features — like edges or corners — in that limited region, dramatically reducing the number of parameters compared to a fully connected layer. Shift‑invariant means those same learned feature detectors (i.e. the same set of weights) are applied across every location in the image. By sharing weights across space, the network recognizes a feature regardless of where it appears — so if an edge or stroke shifts position, it still produces the same response.

#### Core Components
##### Local receptive fields (convolutional layers)
Each hidden unit connects only to a small patch of the previous layer (e.g., a 5×5 window), extracting elementary features like edges or corners at each spatial location .

##### Shared weights (feature maps)
The same filter (set of weights) is applied across all image locations, drastically reducing parameters and enforcing translation invariance. 
Enforcing translation invariance means that features are recognized regardless of where they appear in the image due to shared weights across
all parts of the image.

##### Subsampling layers (pooling)
Periodically the network reduces spatial resolution (e.g., 2×2 average pooling), making the representation more abstract and robust to small shifts/distortions. This helps generalize the model to higher-order features without over-emphasizing on fine-grained details captured in lower-order
features of images.

##### Stacked hierarchy
Lower layers detect simple local patterns; higher layers combine them into increasingly complex, task‑relevant features (from strokes → shapes → digits).

##### Final classifier (RBF output layer)
A small set of class‑specific units score the high‑level features, producing the final digit (or character) prediction.

### Why It Matters
This architecture enables LeNet‑5 to learn powerful, spatially aware representations directly from raw pixels — using far fewer parameters and far less manual feature engineering than a standard MLP — leading to state‑of‑the‑art accuracy on digit recognition with strong robustness to shifts, distortions, and noise. 

### Valuable Insights/Conclusions:
1) **Hierarchical, context‑aware feature learning → Foundation for self‑attention’s contextual embeddings**
LeNet‑5 showed that building layers of increasingly abstract features (edges → shapes → digits) dramatically improves performance on vision tasks, because each layer contextualizes lower‑level patterns into richer concepts. This same principle underlies modern Transformers: instead of treating each token independently, self‑attention learns context‑dependent embeddings by relating every token to its neighbors, capturing hierarchical relationships across sequences.

2) **Weight sharing for translation invariance → Parameter efficiency and generalization**
By forcing identical filters across spatial locations, LeNet‑5 drastically cut its parameter count and achieved shift‑invariance. This insight directly inspired convolutional layers in vision models — and the broader notion of parameter sharing seen in Transformer attention heads, where the same attention mechanism applies across all positions, enabling scalable models that generalize across contexts.

3) **Pooling/subsampling for robustness → Dimensionality reduction in modern architectures**
LeNet‑5’s subsampling layers reduced spatial resolution to build abstract, noise‑resistant representations . Today’s models echo this via pooling in CNNs, strided attention in vision Transformers, and token downsampling in language models — all trading fine detail for efficient, higher‑level feature extraction that improves generalization and computation.

These core LeNet‑5 ideas — hierarchical abstraction, parameter sharing for invariance, and structured dimensionality reduction — form the conceptual backbone of nearly every modern deep architecture, from CNNs to Transformers.