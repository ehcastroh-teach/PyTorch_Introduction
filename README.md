# An Introduction to PyTorch

PyTorch is an open-source machine learning library built on dynamic computation graphs and a NumPy-familiar tensor API. This repository introduces PyTorch from the ground up - covering tensors, automatic differentiation with `autograd`, and building trainable models with `nn.Module` - so that readers can move from raw tensor math to a working training loop without prior machine learning experience.

---

## Learning Objectives

By working through this repository you will be able to:

- Create and manipulate PyTorch tensors across scalar, vector, matrix, and higher-rank shapes
- Use tensor operations (arithmetic, indexing, slicing, reshape) and understand how they differ from NumPy
- Enable gradient tracking with `requires_grad` and compute gradients using `.backward()`
- Build a neural network by subclassing `nn.Module` and implementing `forward`
- Write a complete training loop using a loss function and an optimizer
- Understand when to use `model.eval()` and `torch.no_grad()` during inference

---

## File Dictionary

| File / Folder | Description |
|---|---|
| `pytorch_introduction.ipynb` | Main notebook - four parts covering setup, tensors, autograd, and nn.Module |
| `assets/homeworks/pytorch_introduction_homework.ipynb` | Practice exercises using the California housing dataset |
| `assets/content/images/` | Logo and visual assets used inside the notebook |
| `requirements.txt` | Python package dependencies |

---

## Workflow Diagram

```
Clone repo
    |
    v
Install dependencies (requirements.txt)
    |
    v
Open pytorch_introduction.ipynb
    |
    +---> Part 0: About and Motivation
    |
    +---> Part 1: Tensors and Operations
    |         |
    |         +-- 1.1 PyTorch Setup
    |         +-- 1.2 Tensor Creation (factory functions, NumPy interop)
    |         +-- 1.3 Tensor Operations (arithmetic, reductions, broadcasting)
    |         +-- 1.4 Indexing, Slicing, and Reshaping
    |
    +---> Part 2: Autograd - Automatic Differentiation
    |         |
    |         +-- 2.1 requires_grad and the computation graph
    |         +-- 2.2 Calling .backward()
    |         +-- 2.3 Stopping gradient tracking
    |
    +---> Part 3: Neural Networks with nn.Module
    |         |
    |         +-- 3.1 Building a model with nn.Module
    |         +-- 3.2 The training loop (zero_grad, forward, backward, step)
    |         +-- 3.3 Evaluating the model
    |
    +---> Part 4: Wrap-Up and Next Steps
    |
    +---> Appendix: Installation guide, references
    |
    v
Work through the homework notebook
```

---

## Step-by-Step Walkthrough

### Part 0 - About and Motivation

PyTorch builds its computation graph dynamically as Python executes, which means you can use ordinary Python control flow and inspect tensors at any point during a forward pass. The notebook opens with a side-by-side comparison of PyTorch and TensorFlow 2.x to orient learners who may have seen one framework already.

### Part 1.1 - PyTorch Setup

Installing PyTorch and confirming the version number. The right install command depends on your OS and whether you have a CUDA GPU - the notebook links to the official install selector at pytorch.org rather than hardcoding a wheel that may be outdated.

### Part 1.2 - Tensor Creation

PyTorch tensors are created with factory functions (`torch.tensor`, `torch.zeros`, `torch.randn`, `torch.arange`) rather than class constructors. The section also covers the `torch.from_numpy` / `.numpy()` bridge and explains the shared-memory behavior that makes it efficient but occasionally surprising.

### Part 1.3 - Tensor Operations

Operations in PyTorch follow NumPy conventions, with operator overloading for the common cases (`+`, `*`, `@`). The section distinguishes element-wise multiplication (`torch.mul`) from matrix multiplication (`torch.matmul`) and covers in-place operations (trailing underscore convention, e.g., `add_`).

### Part 1.4 - Indexing, Slicing, and Reshaping

Integer indexing removes a dimension; slice indexing preserves it. `.view()` returns a new shape over the same storage (requires contiguous memory); `.reshape()` handles the non-contiguous case by copying if needed. Using `-1` as a dimension lets PyTorch infer the size.

### Part 2.1 - requires_grad and the Computation Graph

`requires_grad=True` flags a tensor so that every downstream operation records itself in a `grad_fn` chain. This chain is the computation graph; `.backward()` walks it in reverse, accumulating chain-rule products into each leaf's `.grad`.

### Part 2.2 - Calling .backward()

`.backward()` computes gradients for all leaf tensors in the graph. Gradients accumulate across calls - always call `optimizer.zero_grad()` (or `tensor.grad.zero_()`) before computing new gradients in a training loop.

### Part 2.3 - Stopping Gradient Tracking

`torch.no_grad()` disables graph construction for a block of code, reducing memory and compute during inference. `.detach()` creates a tensor that shares storage with the original but is excluded from the graph.

### Part 3.1 - Building a Model with nn.Module

Every custom model subclasses `nn.Module` and implements two methods: `__init__` (define layers) and `forward` (define the computation). PyTorch automatically tracks all parameters registered as attributes, making them available via `model.parameters()` for the optimizer.

### Part 3.2 - The Training Loop

The four-step pattern per batch: `optimizer.zero_grad()` - forward pass - `loss.backward()` - `optimizer.step()`. The example fits a linear model to noisy data drawn from $y = 2x + 1$ and plots the loss curve and learned fit.

### Part 3.3 - Evaluating the Model

`model.eval()` flips behavior flags in layers like Dropout and BatchNorm. `torch.no_grad()` stops graph construction. Both are needed together during evaluation - they solve different problems.

---

## How to Run

```bash
# 1. Clone and enter the repo
git clone https://github.com/ehcastroh-teach/PyTorch_Introduction.git
cd PyTorch_Introduction

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook pytorch_introduction.ipynb
```

Run all cells top-to-bottom (Kernel - Restart & Run All).

---

## Key Concepts Glossary

| Term | Plain-language definition |
|---|---|
| Tensor | A multi-dimensional array - the basic unit of data in PyTorch |
| Rank | The number of dimensions of a tensor (scalar = rank 0, vector = rank 1, matrix = rank 2) |
| `torch.tensor` | Factory function that creates a new tensor from data (lowercase `t`) |
| `requires_grad` | Flag that tells PyTorch to track operations on a tensor for gradient computation |
| `grad_fn` | The function node that produced a tensor; the link in the computation graph |
| `.backward()` | Computes gradients of a scalar output with respect to all leaf tensors in the graph |
| `.grad` | Attribute on a leaf tensor that holds accumulated gradients after `.backward()` |
| `autograd` | PyTorch's automatic differentiation engine |
| `nn.Module` | Base class for all neural network models in PyTorch |
| `forward` | Method on an `nn.Module` that defines the computation for a forward pass |
| `optimizer.zero_grad()` | Clears accumulated gradients before a new backward pass |
| `optimizer.step()` | Updates model parameters using the computed gradients |
| `model.eval()` | Switches layers like Dropout and BatchNorm to inference mode |
| `torch.no_grad()` | Context manager that disables gradient computation (saves memory and compute) |
| Broadcasting | Automatically expanding a smaller tensor to match the shape of a larger one during operations |

---

## Further Reading

- PyTorch Official Tutorials
- Deep Learning with PyTorch (Eli Stevens, Luca Antiga, Thomas Viehmann)
- Programming PyTorch for Deep Learning (Ian Pointer)
- PyTorch Documentation - torch.autograd

---

## Credits and Acknowledgements

Notebook content developed as a complementary introduction to the TensorFlow Introduction repository under ehcastroh-teach. Textbook references: *Deep Learning with PyTorch* (Manning Publications); *Programming PyTorch for Deep Learning* (O'Reilly Media). PyTorch logo copyright PyTorch / Linux Foundation.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
