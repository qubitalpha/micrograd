# Building GPT from Scratch (in Code, Spelled Out)

This directory contains learning notes, code implementations, and Jupyter notebooks for following Andrej Karpathy's **"Let's build GPT: from scratch, in code, spelled out"** tutorial from the *Neural Networks: Zero to Hero* series.

---

## 🎥 References

* **Video Tutorial:** [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
* **Official Google Colab:** [Google Colab Notebook (by Andrej Karpathy)](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-)
* **Companion Repository:** [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

---

## 📚 Notebooks & Progression

### 1. Data & Tokenization
* **[`gpt-dev1.ipynb`](./gpt-dev1.ipynb)**: Input characters, character-level tokenization, train/val split, batch size, and block size context.

### 2. Bigram Language Model Baseline
* **[`gpt-dev2.0.ipynb`](./gpt-dev2.0.ipynb)**: Initial Bigram language model implementation (`nn.Embedding` lookup table).
* **[`gpt-dev2.1.ipynb`](./gpt-dev2.1.ipynb)**: Adding cross-entropy loss function to the bigram language model.
* **[`gpt-dev2.12.ipynb`](./gpt-dev2.12.ipynb)**: Reshaping logits dimensions from $(B, T, C)$ to $(B \cdot T, C)$ and targets to $(B \cdot T)$ to satisfy PyTorch's `F.cross_entropy`.
* **[`gpt-dev2.13.ipynb`](./gpt-dev2.13.ipynb)**: Sampling and generating characters from the untrained bigram model.
* **[`gpt-dev2.14.ipynb`](./gpt-dev2.14.ipynb)**: Training the bigram neural network with `torch.optim.AdamW`.
* **[`gpt-dev2.15.ipynb`](./gpt-dev2.15.ipynb)**: Complete refactored baseline bigram language model script (`bigram.py`).

### 3. Attention & Transformer Architecture
* **[`gpt-dev3.0.ipynb`](./gpt-dev3.0.ipynb)**: Baseline transition from Bigram to self-attention.
* **[`gpt-dev3.1.ipynb`](./gpt-dev3.1.ipynb)**: The mathematical trick in self-attention (averaging past tokens via matrix multiplication and Softmax).
* **[`gpt-dev3.2 Attention and self-attention.ipynb`](./gpt-dev3.2%20Attention%20and%20self-attention.ipynb)**: Query, Key, and Value vectors; scaled dot-product attention head.
* **[`gpt-dev3.3 self-attention for a spin.ipynb`](./gpt-dev3.3%20self-attention%20for%20a%20spin.ipynb)**: Plugging the single self-attention head into the language model.
* **[`gpt-dev4 multi-head attention.ipynb`](./gpt-dev4%20multi-head%20attention.ipynb)**: Multi-Head Attention (running multiple attention heads in parallel).
* **[`gpt-dev5 feed forward.ipynb`](./gpt-dev5%20feed%20forward.ipynb)**: Position-wise Feed-Forward networks (per-token MLP computation).
* **[`gpt-dev6 implementing blocks.ipynb`](./gpt-dev6%20implementing%20blocks.ipynb)**: Combining Multi-Head Attention and Feed-Forward into Transformer Blocks.
* **[`gpt-dev7 residual pathways.ipynb`](./gpt-dev7%20residual%20pathways.ipynb)**: Adding residual connections (skip connections) around sub-layers.
* **[`gpt-dev8 Layer Norm.ipynb`](./gpt-dev8%20Layer%20Norm.ipynb)**: Pre-layer normalization (`nn.LayerNorm`) and final scaling.

### Reference Files
* **[`andrej_gpt_colab.ipynb`](./andrej_gpt_colab.ipynb)**: Exact local copy of Andrej's finished Colab notebook including all intermediate explanations, experiments, and reference cells.
* **`input.txt`**: Tiny Shakespeare dataset (~1.1MB text) used for training the character-level model.

---

## 🛠️ Environment & Kernel

* **Python:** 3.9+ (via `micrograd/venv`)
* **Jupyter Kernel:** `Python (micrograd)`
* **Dependencies:** `torch`, `matplotlib`
