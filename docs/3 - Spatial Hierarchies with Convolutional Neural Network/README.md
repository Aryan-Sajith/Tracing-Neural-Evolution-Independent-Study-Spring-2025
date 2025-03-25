## Brief:

Here we explore a key enhancement to the multi-layer perceptron discussed in the last section-- a convolutional neural network designed specifically to capture essential spatial hierarchies of information within images.

# Paper 1: [Gradient-Based Learning Applied to Document Recognition](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf)

- **Date Published:** 1998
- **Authors** - Yann Lecunn, Leon Bottou, Yoshua Bengio, Patrick Haffner

### Key Idea:

Baseline MLPs struggle with capturing spatial hierarchies of information(increasingly complex representations that feed into each other like colors/edges -> shapes/forms -> people/animals) due to them treating all pixels as essentially structurally independent and lacking proper scaling when dealing with even relatively large images(eg: 40 x 40 pixel images and 100 hidden nodes = 40 x 40 x 100 = 160,000 weights!) due to learning weights for each pixel. Convolutional neural networks tackle this by using local receptive fields(compressional techniques that capture context over chunks of images as opposed to dealing with all pixels individually), replicated(shared) weights across increasing layers of abstraction, and subsampling(the representation of knowledge in an increasingly abstracted fashion where lower levels capture lower-order features and use that as a base to capture higher order features in subsequent layers).