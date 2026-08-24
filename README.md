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
