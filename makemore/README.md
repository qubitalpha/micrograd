# Makemore Learning Notes & Notebooks

This directory contains notes, code snippets, and Jupyter notebooks for **makemore**—inspired by Andrej Karpathy's neural network series.

The goal of this module is to build character-level language models from scratch, starting with bigram counting statistics and progressing through PyTorch tensor operations and neural network optimization.

---

## 🎥 References

* **Video Tutorial:** [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo?si=JgsNco-makH8eLcm)

---

## 📚 Learning Progression & Notebooks

You can follow along with the notebooks in this exact chronological order:

1. **[makemore-1.ipynb](./makemore-1.ipynb)**
   * **Topics:** Introduction to character-level bigram language models, exploring the `names.txt` dataset, pairing consecutive characters, and counting frequencies with Python dictionaries.
   * **Key Takeaway:** Understanding how bigram pairs capture character transition probabilities (e.g., start/end characters and common naming patterns).

2. **[makemore-2.ipynb](./makemore-2.ipynb)**
   * **Topics:** Representing bigram counts in a $27 \times 27$ PyTorch tensor, mapping characters to integers (`stoi` and `itos`), probability normalization, sampling new names from the learned distribution, and evaluating model quality with negative log-likelihood (NLL).
   * **Key Takeaway:** Structuring frequency counts into matrix operations and measuring model performance using average loss.

3. **[makemore-3.ipynb](./makemore-3.ipynb)**
   * **Topics:** Tensor broadcasting semantics, probability smoothing, calculating cross-entropy loss, and transitioning to a single-layer neural network using one-hot encoded character inputs and PyTorch autograd.
   * **Key Takeaway:** Seeing how a neural network optimized via gradient descent reproduces identical probability estimates to direct count normalization.

---

## 🛠️ Environment Setup & Dependencies

* **Python:** 3.9+
* **Key Libraries:** `torch`, `matplotlib`, `jupyter`
* **Dataset:** `names.txt` (included in `makemore/`)
