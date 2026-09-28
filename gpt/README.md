# Building GPT from Scratch (in Code, Spelled Out)

This directory contains learning notes, code implementations, and Jupyter notebooks documenting the step-by-step reproduction of a Generative Pre-trained Transformer (GPT) language model from scratch, inspired by Andrej Karpathy's lecture in the *Neural Networks: Zero to Hero* series.

---

## 🎥 References

* **Video Lecture:** [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
* **Official Google Colab:** [Google Colab Notebook (by Andrej Karpathy)](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-)
* **Companion Repository:** [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)
* **Foundational Paper:** [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## 📚 Notebook Summaries & Progression

### Phase 1: Data, Tokenization & Context Dynamics

#### 📓 [`gpt-dev1.ipynb`](./gpt-dev1.ipynb) — Data Preparation & Context Chunking
* **Core Topics:**
  * Loading and inspecting the raw Tiny Shakespeare dataset (`1,115,394` characters).
  * Building a character-level vocabulary (`vocab_size = 65`) and lookup tables (`stoi`, `itos`).
  * Contrast between simple character-level tokenization and subword BPE (Byte Pair Encoding) via `tiktoken` (vocab size `50,257`).
  * Splitting dataset into training (`90%` = $1{,}003{,}854$ characters) and validation (`10%` = $111{,}540$ characters).
  * Understanding `block_size` (context length): how a sequence of length $T$ simultaneously yields $T$ distinct input-target prediction examples.
  * Understanding `batch_size`: stacking independent chunks in parallel for GPU compute efficiency without cross-sequence interaction.

---

### Phase 2: Building the Bigram Neural Network Baseline

#### 📓 [`gpt-dev2.0.ipynb`](./gpt-dev2.0.ipynb) — Bigram Architecture Skeleton
* **Core Topics:**
  * Implementing the simplest neural language model: `BigramLanguageModel(nn.Module)`.
  * Using `nn.Embedding(vocab_size, vocab_size)` as a direct lookup table where the embedding vector for each token directly represents the raw output logits for the next token.
  * Baseline forward pass without loss computation.

#### 📓 [`gpt-dev2.1.ipynb`](./gpt-dev2.1.ipynb) — Attempting Cross-Entropy Loss
* **Core Topics:**
  * Introducing `F.cross_entropy(logits, targets)` into the model forward pass.
  * Identifying the tensor dimension mismatch: PyTorch expects channels as the 2nd dimension for multi-dimensional inputs (shape $(B, C, T)$) or a 2D tensor $(N, C)$, while raw model output is $(B, T, C)$ with targets of shape $(B, T)$.

#### 📓 [`gpt-dev2.12.ipynb`](./gpt-dev2.12.ipynb) — Logits Reshaping & Initial Loss Analysis
* **Core Topics:**
  * Resolving the dimension mismatch by flattening the batch and time dimensions: `logits.view(B*T, C)` and `targets.view(B*T)`.
  * Measuring the initial model loss ($\approx 4.88$).
  * Comparing against the theoretical baseline for uniform random chance: $-\ln(1/65) \approx 4.17$.
  * Recognizing that an initial loss higher than uniform chance indicates suboptimal parameter initialization (initial weights predict with negative bias rather than diffuse uncertainty).

#### 📓 [`gpt-dev2.13.ipynb`](./gpt-dev2.13.ipynb) — Autoregressive Generation Baseline
* **Core Topics:**
  * Implementing the `generate(idx, max_new_tokens)` method: looping forward passes, extracting last-step logits, computing softmax probabilities, and sampling via `torch.multinomial`.
  * Sampling text from the untrained model to establish the random baseline output ("garbage" text).

#### 📓 [`gpt-dev2.14.ipynb`](./gpt-dev2.14.ipynb) — Optimization with AdamW
* **Core Topics:**
  * Introducing the training loop with `torch.optim.AdamW(m.parameters(), lr=1e-3)`.
  * Batch training over multiple steps, computing loss, zeroing gradients, backpropagating, and stepping optimizer.
  * Observing loss reduction and noticing emerging pseudo-word structure in generation, while acknowledging that tokens still cannot communicate across context history.

#### 📓 [`gpt-dev2.15.ipynb`](./gpt-dev2.15.ipynb) — Standalone Bigram Script (`bigram.py`)
* **Core Topics:**
  * Consolidating all bigram experiments into a clean, reproducible script format matching Karpathy's `bigram.py`.
  * Implementing `estimate_loss()` across train and validation sets with `@torch.no_grad()`.

---

### Phase 3: The Transformer Architecture & Self-Attention

#### 📓 [`gpt-dev3.0.ipynb`](./gpt-dev3.0.ipynb) — The Mathematical Trick in Attention
* **Core Topics:**
  * Motivation: How can previous tokens aggregate information into the current position without lookahead bias?
  * Using matrix multiplication as a weighted aggregation mechanism.
  * Implementing lower triangular matrices (`torch.tril`) and normalizing rows with `torch.sum(dim=1, keepdim=True)` to calculate causal historical averages.

#### 📓 [`gpt-dev3.1.ipynb`](./gpt-dev3.1.ipynb) — Positional Embeddings
* **Core Topics:**
  * Attention is permutation-invariant: unlike RNNs or CNNs, pure attention has no inherent sense of sequence order.
  * Adding a learnable `position_embedding_table` (`nn.Embedding(block_size, n_embd)`).
  * Conditioning input representations on both content and location: $x = \text{tok\_emb} + \text{pos\_emb}$.

#### 📓 [`gpt-dev3.2 Attention and self-attention.ipynb`](./gpt-dev3.2%20Attention%20and%20self-attention.ipynb) — Scaled Dot-Product Self-Attention
* **Core Topics:**
  * Deriving the Self-Attention mechanism: Queries, Keys, and Values.
    * **Query ($Q$):** What the current token is looking for.
    * **Key ($K$):** What the current token contains.
    * **Value ($V$):** The actual communicated payload information.
  * Calculating affinity scores: $wei = Q \cdot K^T / \sqrt{d_k}$.
  * Scaling by $1/\sqrt{d_k}$ to prevent variance explosion and preserve stable softmax gradients.
  * Causal masking: setting future token affinities to $-\infty$ prior to softmax.
  * Aggregating values: $out = wei \cdot V$.
  * Viewing attention as a data-dependent, directed communication graph between tokens.

#### 📓 [`gpt-dev3.3 self-attention for a spin.ipynb`](./gpt-dev3.3%20self-attention%20for%20a%20spin.ipynb) — Integrating Single-Head Attention
* **Core Topics:**
  * Embedding the single `Head` module into the language model architecture.
  * Observing loss reduction compared to the bigram baseline as tokens now communicate across sequence context.

#### 📓 [`gpt-dev4 multi-head attention.ipynb`](./gpt-dev4%20multi-head%20attention.ipynb) — Multi-Head Attention
* **Core Topics:**
  * Implementing `MultiHeadAttention`: running multiple attention heads in parallel with smaller head dimension ($d_{\text{head}} = n_{\text{embd}} / n_{\text{head}}$).
  * Allowing tokens to look for multiple distinct types of relationships simultaneously (e.g., consonant-vowel patterns, punctuation, semantic context).
  * Concatenating head outputs and passing through a final linear projection layer.

#### 📓 [`gpt-dev5 feed forward.ipynb`](./gpt-dev5%20feed%20forward.ipynb) — Position-Wise Feed-Forward Network
* **Core Topics:**
  * Differentiating between "Communication" (Self-Attention) and "Computation" (Feed-Forward).
  * Adding a per-token multi-layer perceptron (`FeedForward`): Linear $\to$ ReLU $\to$ Linear.
  * Allowing each token to independently process and transform the information gathered from other tokens.

#### 📓 [`gpt-dev6 implementing blocks.ipynb`](./gpt-dev6%20implementing%20blocks.ipynb) — Transformer Blocks
* **Core Topics:**
  * Encapsulating `MultiHeadAttention` and `FeedForward` into a repeatable `Block` module.
  * Stacking multiple transformer blocks sequentially (`nn.Sequential(*[Block(...) for _ in range(n_layer)])`).
  * Observing that deeper networks become harder to optimize without residual connections.

#### 📓 [`gpt-dev7 residual pathways.ipynb`](./gpt-dev7%20residual%20pathways.ipynb) — Residual Connections
* **Core Topics:**
  * Implementing residual (skip) connections around each sub-layer:
    $$x = x + \text{sa\_heads}(x)$$
    $$x = x + \text{ffwd}(x)$$
  * Based on He et al. (ResNet, 2015): creating an unimpeded gradient highway directly from output to input during backpropagation.
  * Adding projection layers with zero/near-zero initializations to preserve the residual stream.

#### 📓 [`gpt-dev8 Layer Norm.ipynb`](./gpt-dev8%20Layer%20Norm.ipynb) — Pre-Layer Normalization
* **Core Topics:**
  * Implementing Layer Normalization (`nn.LayerNorm`, Ba et al., 2016): normalizing across channel dimensions independently per token.
  * Adopting modern Pre-Norm architecture (normalizing input *before* entering Attention and Feed-Forward sub-layers, rather than Post-Norm from original Vaswani et al. 2017).
  * Adding final layer normalization (`ln_f`) before the language model projection head (`lm_head`).

---

### Reference Artifacts

* **[`andrej_gpt_colab.ipynb`](./andrej_gpt_colab.ipynb)**: Full, exact local copy of Andrej Karpathy's completed reference notebook from the video description (all 39 cells, outputs, and mathematical diagrams).
* **[`input.txt`](./input.txt)**: Tiny Shakespeare dataset (~1.1MB text) used as the training corpus.

---

## 🛠️ Environment & Kernel

* **Python Version:** 3.9+ (via `micrograd/venv`)
* **Jupyter Kernel:** `Python (micrograd)`
* **Dependencies:** `torch`, `matplotlib`, `tiktoken`, `transformers`
