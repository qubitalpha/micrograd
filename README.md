# Micrograd & Makemore Learning Repository

This repository contains learning notes, code implementations, and Jupyter notebooks exploring neural network mechanics and language modeling from scratch—inspired by Andrej Karpathy's tutorials (**micrograd** and **makemore**).

---

## 📂 Modules & Notebook Indexes

### 1. 🧠 [Backpropagation & Micrograd](./backpropagation/README.md)
Foundational understanding of automatic differentiation, scalar `Value` engine, Multi-Layer Perceptrons (MLPs), loss functions, and backpropagation from scratch.

* **[micrograd_basic_forward_and_backprop.ipynb](./backpropagation/micrograd_basic_forward_and_backprop.ipynb)**: Core `Value` engine, forward pass, and manual backpropagation.
* **[micrograd_breaking_tanh.ipynb](./backpropagation/micrograd_breaking_tanh.ipynb)**: Non-linear activation functions (`tanh`) and numerical stability.
* **[micrograd_MLP_basics.ipynb](./backpropagation/micrograd_MLP_basics.ipynb)**: Object-oriented MLP architecture (`Value` → `Neuron` → `Layer` → `MLP`).
* **[micrograd_MLP_loss_func_intro.ipynb](./backpropagation/micrograd_MLP_loss_func_intro.ipynb)**: Loss functions (MSE / max-margin) and optimization targets.
* **[micrograd_successful_NN_training.ipynb](./backpropagation/micrograd_successful_NN_training.ipynb)**: Complete training loop and gradient descent optimization.

### 2. 🔤 [Makemore & Language Models](./makemore/README.md)
Character-level language modeling from bigram statistics to PyTorch neural network formulations.

* **[makemore-1.ipynb](./makemore/makemore-1.ipynb)**: Bigram counting intro and character transition pairs.
* **[makemore-2.ipynb](./makemore/makemore-2.ipynb)**: PyTorch tensor matrix representation, sampling, and negative log-likelihood (NLL).
* **[makemore-3.ipynb](./makemore/makemore-3.ipynb)**: Broadcasting semantics, loss calculation, and bigram neural network optimization via PyTorch autograd.
* **[makemore-4.ipynb](./makemore/makemore-4.ipynb)**: Neural network formulation, one-hot encoding, Softmax, and loss minimization.
* **[makemore-5.ipynb](./makemore/makemore-5.ipynb)**: Complete training loop and sampling names from the trained neural network.

### 3. 🤖 [Building GPT from Scratch](./gpt/README.md)
Generative Pre-trained Transformer (GPT) language model from scratch in PyTorch—covering bigram baselines, scaled dot-product attention, multi-head attention, feed-forward layers, residual pathways, and layer normalization.

* **[gpt-dev1.ipynb](./gpt/gpt-dev1.ipynb)**: Data loading, character-level tokenization, train/val split, batch & block size context.
* **[gpt-dev2.0.ipynb](./gpt/gpt-dev2.0.ipynb)**: Baseline bigram neural network architecture (`nn.Embedding` lookup table).
* **[gpt-dev2.1.ipynb](./gpt/gpt-dev2.1.ipynb)**: Adding cross-entropy loss function to the bigram language model.
* **[gpt-dev2.12.ipynb](./gpt/gpt-dev2.12.ipynb)**: Reshaping tensor dimensions for `F.cross_entropy` and initial loss baseline analysis.
* **[gpt-dev2.13.ipynb](./gpt/gpt-dev2.13.ipynb)**: Autoregressive sampling and generating from the untrained model.
* **[gpt-dev2.14.ipynb](./gpt/gpt-dev2.14.ipynb)**: Training loop optimization with AdamW and sampling from trained bigram.
* **[gpt-dev2.15.ipynb](./gpt/gpt-dev2.15.ipynb)**: Consolidated standalone baseline script (`bigram.py`).
* **[gpt-dev3.0.ipynb](./gpt/gpt-dev3.0.ipynb)**: Mathematical trick in self-attention (causal averaging with `tril` and matrix multiplication).
* **[gpt-dev3.1.ipynb](./gpt/gpt-dev3.1.ipynb)**: Positional embeddings ($x = \text{tok\_emb} + \text{pos\_emb}$) for sequence order.
* **[gpt-dev3.2 Attention and self-attention.ipynb](./gpt/gpt-dev3.2%20Attention%20and%20self-attention.ipynb)**: Queries, Keys, Values, causal masking, and scaled dot-product attention head.
* **[gpt-dev3.3 self-attention for a spin.ipynb](./gpt/gpt-dev3.3%20self-attention%20for%20a%20spin.ipynb)**: Integrating the single attention head into the model.
* **[gpt-dev4 multi-head attention.ipynb](./gpt/gpt-dev4%20multi-head%20attention.ipynb)**: Multi-head attention running multiple heads in parallel.
* **[gpt-dev5 feed forward.ipynb](./gpt/gpt-dev5%20feed%20forward.ipynb)**: Position-wise feed-forward networks (per-token MLP computation).
* **[gpt-dev6 implementing blocks.ipynb](./gpt/gpt-dev6%20implementing%20blocks.ipynb)**: Transformer blocks stacking attention and feed-forward.
* **[gpt-dev7 residual pathways.ipynb](./gpt/gpt-dev7%20residual%20pathways.ipynb)**: Residual connections (skip connections) around sub-layers.
* **[gpt-dev8 Layer Norm.ipynb](./gpt/gpt-dev8%20Layer%20Norm.ipynb)**: Pre-layer normalization (`nn.LayerNorm`) and final scaling.
* **[gpt.py (Andrej's Full Reference Script)](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)**: Full flushed-out Transformer implementation. *(Note: This configuration cannot practically run on a MacBook—it requires a GPU, or a deeper understanding to break down and scale down the layers, heads, and embedding dimensions).*

---

## 🛠️ Quick Start

```bash
# Clone repository
git clone https://github.com/qubitalpha/micrograd.git
cd micrograd

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install jupyter graphviz torch torchviz matplotlib

# Launch Jupyter Notebooks
jupyter notebook
```
