# 08. Evaluation Pipeline

## Evaluation Metrics
1. **Perplexity (PPL)**: Exponential cross-entropy loss measuring language modeling fluency on unseen test sets.
2. **ROUGE Metrics**: Computes ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (longest common subsequence) overlap between generated outputs and ground truth references.
3. **BLEU Score**: Calculates n-gram precision (BLEU-1, BLEU-2, BLEU-4) with brevity penalty.
4. **Exact Match (EM)**: Verifies exact token string matching for structured prediction or code tasks.
5. **Side-by-Side Playground**: Allows human side-by-side evaluation of base model vs fine-tuned model outputs across prompt benchmarks.
