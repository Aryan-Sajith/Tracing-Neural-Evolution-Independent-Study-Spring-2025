
# Multilayer Perceptron (MLP) — Poor Performance on CIFAR‑10

**Model Type:**  
Multilayer Perceptron classifier (scikit‑learn).

**Task:**  
CIFAR‑10 image classification (10 classes; 32×32 color images).

**Cons of MLP:**  
MLPs struggle with certain complex image‑classification tasks due to the following reasons:
- **Loss of Spatial Information:** MLPs do not inherently understand spatial relationships in images. 
Meaning that they do not understand that pixels close to each other are more related than pixels far apart,
this lack of context makes it difficult for MLPs to learn spatial features like edges, textures, and shapes.
This is a critical limitation for more complex image classification tasks beyond MNIST.
- **Parameteric Efficiency:** MLPs require a large number of parameters to learn spatial features.
This is because each neuron in the first hidden layer must learn a separate weight for each pixel in the input image whereas
you can share weights across abstracted image regions to more efficiently learn spatial features. For a simple example,
instead of learning a separate weight for each pixel in a 28x28 image, you can learn a single weight for each 2x2 region.
- **Lack of Hierarchical Feature Learning:** MLPs do not learn hierarchical features.
In image classification, features are often hierarchical, where lower layers learn basic features like edges and textures,
and higher layers learn more complex features like shapes and objects. Simple MLPs do not have this hierarchical feature learning capability 
since they treat all input features as structurally equivalent. Instead of this, you can learn hierarchical features by 
using abstracted representations of image features in increasing order of complexity. For a simple example, you can use a pooling layer to 
abstractly represent edge features learned in the first hidden layer, and then use this abstracted representation as input to the second hidden layer.
Such hierarchical feature learning allows you to learn more complex features with fewer parameters and less data. This is precisely what 
convolutional neural networks (CNNs) do.

## Test Summary & Results for MLP:
- **Accuracy:** 0.429  
- **Precision (weighted):** 0.435  
- **Recall (weighted):** 0.429  
- **F1 Score (weighted):** 0.430

As can be seen from the results, the MLP struggles to map the CIFAR‑10 images to their correct classes.
This demonstrates that a baseline fully‑connected MLP struggles on natural‑image classification tasks that require spatial understanding 
and hierarchical feature learning. To address these limitations, we can use convolutional neural networks (CNNs) which are specifically
designed to learn spatial features and hierarchical representations in images. To see the benefits of CNNs, refer to the output/cnn/good-cnn-test.md file.
