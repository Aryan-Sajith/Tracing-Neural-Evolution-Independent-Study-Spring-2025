## Brief:

Here we explore a seminal advancement in neural networks designed to tackle sequential processing tasks like text parsing/generation. We use
Long Short‑Term Memory (LSTM), a landmark recurrent architecture that overcomes the vanishing‑gradient limitations of standard RNNs to learn extremely long‑range dependencies in sequential tasks such as text parsing and generation .


## Paper 1: [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

- **Date Published:** 1997
- **Authors** - Sepp Hochreiter, Jurgen Schmidhuber

## Key Idea:
Conventional feed‑forward networks treat each token independently and therefore cannot capture context across a sequence. While traditional recurrent neural networks (RNNs) introduce recurrence to model sequential dependencies, they suffer from exponentially vanishing or exploding gradients, preventing them from learning dependencies beyond a few dozen time steps. Long Short‑Term Memory (LSTM) resolves this by embedding a specialized memory cell—the constant error carousel (CEC)—coupled with multiplicative input and output gates that learn when to write, retain, or read information. This gating mechanism enforces constant error propagation through arbitrarily long intervals, enabling effective learning of very long‑range dependencies in sequential data.

