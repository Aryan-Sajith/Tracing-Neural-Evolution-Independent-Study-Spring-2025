
# Standard Transformer (t5-small) — Arithmetic Reasoning (Bad Test)

**Model Type:**
Standard Sequence-to-Sequence Transformer (t5-small) fine-tuned on synthetic arithmetic problems. Intended as a "bad test" baseline.

**Task:**
Solve simple arithmetic problems presented in natural language text.
Ex: "Question: What is 5 plus 3? Answer:" -> Expected Output: "8"

**Cons (Why a standard Transformer struggles here):**
- **Lack of Explicit Reasoning:** Standard Seq2Seq models primarily rely on pattern matching and statistical correlations learned during pre-training and fine-tuning. They lack built-in mechanisms for performing reliable multi-step calculations or symbolic manipulation.
- **Sensitivity to Phrasing:** Performance can degrade significantly with slight changes in wording or the presence of distractors, as the model hasn't learned the underlying arithmetic *procedure*.
- **Limited Generalization:** May memorize answers for patterns seen during training but fails to generalize to unseen number combinations or slightly different problem structures. Requires vast amounts of data to approximate robust reasoning.

**Experiment Setup:**
- Dataset: Synthetic Arithmetic Problems (3000 total, 500 test)
- Max Input Length: 64, Max Target Length: 8
- Epochs: 3
- Batch Size: 16
- Optimizer: Adam (LR=5e-05)

## Test Summary (Exact Match)
- **Accuracy:** 0.024
- **Precision (EM):** 0.024
- **Recall (EM):** 0.024
- **F1 Score (EM):** 0.024
*(Note: Precision/Recall/F1 based on Exact Match)*

**Analysis:**
The standard t5-small Transformer achieves poor accuracy (exact match) on this arithmetic reasoning task, especially given the limited training (3 epochs). It struggles to consistently produce the correct numerical result, likely making errors based on surface-level text patterns rather than performing the actual calculation. This highlights the limitations of standard architectures for tasks requiring explicit, reliable, step-by-step reasoning.

## Next Steps
This poor performance sets the stage for exploring models or techniques designed for reasoning. A model employing Chain-of-Thought (CoT) prompting or fine-tuning would be expected to perform significantly better on this task by explicitly generating the intermediate reasoning steps (e.g., "5 + 3 = 8") before outputting the final answer. Comparing this result to a future CoT implementation would demonstrate the value of explicit reasoning mechanisms.
