# Building GPT from Scratch (in Code, Spelled Out)

This directory contains learning notes, code implementations, and Jupyter notebooks for following Andrej Karpathy's **"Let's build GPT: from scratch, in code, spelled out"** tutorial from the *Neural Networks: Zero to Hero* series.

---

## 🎥 References

* **Video Tutorial:** [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
* **Official Google Colab:** [Google Colab Notebook (by Andrej Karpathy)](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-)
* **Companion Repository:** [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

---

## 📚 Notebooks & Files

* **[`gpt-dev1.ipynb`](./gpt-dev1.ipynb)**: Input characters, character-level tokenization, batch size, and block size.
* **[`gpt-dev2.ipynb`](./gpt-dev2.ipynb)**: Bigram language model implementation without loss function.
* **[`gpt-dev2.1.ipynb`](./gpt-dev2.1.ipynb)**: Adding cross-entropy loss function to the bigram language model.
* **[`gpt-dev2.12.ipynb`](./gpt-dev2.12.ipynb)**: Reshaping logits dimensions from $(B, T, C)$ to $(B \cdot T, C)$ and targets to $(B \cdot T)$ to satisfy PyTorch's `F.cross_entropy`.
* **[`andrej_gpt_colab.ipynb`](./andrej_gpt_colab.ipynb)**: Exact local copy of Andrej's finished Colab notebook including all intermediate explanations, experiments, and reference cells.
* **`input.txt`**: Tiny Shakespeare dataset (~1.1MB text) used for training the character-level model.

---

## 🛠️ Environment & Kernel

* **Python:** 3.9+ (via `micrograd/venv`)
* **Jupyter Kernel:** `Python (micrograd)`
* **Dependencies:** `torch`, `matplotlib`

