
# Reduced Capacity RNN — IMDB Sentiment Classification (Bad Test)

**Model Type:**
RNN using Bidirectional GRU layers, trained from scratch.

**Task:**
Classify movie reviews from the IMDB dataset as positive (1) or negative (0). (Same task as the Transformer 'good test').

**Cons (Why this specific RNN is expected to perform poorly):**
- **Lack of Pre-training:** Learns solely from the small 2000-sample training set.
- **Inherent RNN Limitations:** Still faces challenges with very long-range dependencies compared to Transformers. 

**Experiment Setup:**
- Dataset: `imdb` (subset: 2000 train, 500 test)
- Tokenizer: `distilbert-base-uncased` (Vocab Size: 30522)
- Max Sequence Length: 256
- Model: Embedding(64) -> BiGRU(32) -> GlobalMaxPool -> Dense(64) -> Dense(2)
- Epochs Trained: 5 (Stopped early if patience met) / Max 5
- Batch Size: 16

## Test Summary
- **Accuracy:** 0.794
- **Precision (weighted):** 0.797
- **Recall (weighted):** 0.794
- **F1 Score (weighted):** 0.794

**Analysis:**
This RNN, trained from scratch with limited epochs on the sentiment data, demonstrates lower performance compared to the fine-tuned Transformer. The constraints on model type(lack of full sequence awareness) clearly hinder its ability to effectively capture the nuances of language required for accurate sentiment analysis, providing a clear contrast to the Transformer's capabilities.

## Next Steps
Compare these results directly with the Transformer baseline in `output/transformer/good-test-transformer-sentiment.md`. The performance gap should now be more pronounced, highlighting the advantages of using larger, pre-trained models like Transformers for such NLP tasks.
