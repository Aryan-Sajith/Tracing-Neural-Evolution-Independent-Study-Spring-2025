# Chain-of-Thought (CoT) Neural Network Architectures

## Brief
Chain-of-Thought (CoT) architectures augment standard Transformer-based language models with explicit intermediate reasoning steps—either via prompting or fine-tuning—to break down complex problems into sub-problems. First popularized by Wei et al.’s “Chain of Thought Prompting” (2022), these methods yield dramatic gains on multi-step reasoning benchmarks by mimicking human step-by-step problem solving.

## Paper 1: [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- **Date Published:** January 2022 (ICLR Workshop)  
- **Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc Le, Denny Zhou

## Paper 2: [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
- **Date Published:** May 2022 (NeurIPS ’22)  
- **Authors:** Wenhu Chen, Xueguang Ma, Eric Crawford, Dmitry Lepikhin, Sachin Mehta, Colin Raffel, Raman Chadha

---

## Key Idea
Instead of predicting the final answer in one shot, CoT methods prompt or train the model to generate a **chain of intermediate reasoning steps**. These chains serve both to scaffold the model’s thought process and to expose each sub-step for human inspection or automatic verification.

---

## Methodological Essentials

### 1. Prompt-Based Chain-of-Thought
- **Few-Shot Exemplars**  
  Provide several Q&A pairs where each answer includes a detailed reasoning chain.
- **“Let’s think step by step” Trigger**  
  Appending this phrase elicits a breakdown of the model’s reasoning before the answer.

### 2. Least-to-Most Prompting (L2M)
1. **Decompose** the main problem into an ordered list of simpler sub-questions.  
2. **Solve** each sub-question in sequence, conditioning on prior answers.  
3. **Compose** the final answer from the chain of sub-solutions.

### 3. Supervised CoT Fine-Tuning
- Train a model on datasets annotated with gold reasoning chains (e.g., GSM8K arithmetic tasks).

### 4. Self-Consistency Decoding
- Sample multiple CoT outputs at inference.  
- Aggregate the most consistent final answers across samples.

### 5. Architectural Compatibility
- All CoT techniques leverage standard Transformer decoders (e.g., GPT-style).  
- No new model layers are required—CoT is driven by **data** and **prompt design**.

---

## Why It Matters
1. **Large Accuracy Gains**  
   CoT prompting often doubles or triples performance on multi-step benchmarks (e.g., GSM8K, AQUA-RAT).  
2. **Interpretability**  
   Generated chains allow practitioners to audit and correct reasoning failures.  
3. **Scalability**  
   Chain-of-Thought benefits scale with model size—emergent reasoning appears in models with ≥10 B parameters.

---

## Valuable Insights / Conclusions
1. **Step-wise Deliberation**  
   Breaking down problems mirrors human cognition and reduces “short-circuit” guessing.  
2. **Scale & Reasoning Synergy**  
   Larger models produce richer, more accurate reasoning chains.  
3. **Prompt Quality is Key**  
   Well-crafted exemplars and decomposition strategies can rival or exceed supervised fine-tuning on many tasks.  
```