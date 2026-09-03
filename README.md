# An Introduction to PyTorch

PyTorch is an open-source machine learning library built on dynamic computation graphs and a NumPy-familiar tensor API. This repository introduces PyTorch from the ground up - covering tensors, automatic differentiation with `autograd`, and building trainable models with `nn.Module` - so that readers can move from raw tensor math to a working training loop without prior deep learning experience. Prerequisites are Python, NumPy, and basic linear algebra.

---

## Learning Objectives

By working through this repository you will be able to:

- Create and manipulate PyTorch tensors across scalar, vector, matrix, and higher-rank shapes
- Use tensor operations (arithmetic, indexing, slicing, reshape) and understand how they differ from NumPy
- Explain why `torch.from_numpy()` shares memory and when to use `.clone()` to break that link
- Enable gradient tracking with `requires_grad` and compute gradients using `.backward()`
- Interpret the `grad_fn` chain as a record of the forward computation
- Build a neural network by subclassing `nn.Module` and implementing `__init__` and `forward`
- Write a complete four-step training loop using a loss function and an optimizer
- Understand when to use `model.eval()` and `torch.no_grad()` during inference and why each is needed

---

## Data / File Dictionary

| File / Folder | Description |
|---|---|
| `pytorch_introduction.ipynb` | Main notebook - four parts covering PyTorch setup, tensors, autograd, and nn.Module |
| `assets/homeworks/pytorch_introduction_homework.ipynb` | Practice exercises: tensor ops, gradient computation, and fitting an MLP on the California housing dataset |
| `assets/content/images/pytorch_logo.png` | PyTorch logo used inside the notebook |
| `images/thumbnails/` | Banner images used in the contact block |
| `requirements.txt` | Python package dependencies (torch, numpy, matplotlib, scikit-learn, jupyter) |

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
    |         - PyTorch vs TensorFlow 2.x comparison table
    |         - Why dynamic computation graphs matter for debugging
    |
    +---> Part 1: Tensors and Operations
    |         |
    |         +-- 1.1 PyTorch Setup (version check, device detection)
    |         +-- 1.2 Tensor Creation (factory functions, NumPy interop, shared memory)
    |         +-- 1.3 Tensor Operations (arithmetic, reductions, broadcasting, in-place)
    |         +-- 1.4 Indexing, Slicing, and Reshaping (.view vs .reshape, -1 inference)
    |         +-- Concept Check: rank-3 tensor indexing and reshape
    |
    +---> Part 2: Autograd - Automatic Differentiation
    |         |
    |         +-- 2.1 requires_grad and the computation graph (grad_fn chain)
    |         +-- 2.2 Calling .backward() (single and multi-variable examples)
    |         +-- 2.3 Stopping gradient tracking (no_grad vs detach)
    |         +-- Concept Check: analytical vs autograd gradient comparison
    |
    +---> Part 3: Neural Networks with nn.Module
    |         |
    |         +-- 3.1 Building a model (nn.Module subclass and nn.Sequential)
    |         +-- 3.2 The training loop (zero_grad, forward, backward, step)
    |         +-- 3.3 Evaluating the model (eval mode + no_grad together)
    |         +-- Concept Check: two-layer MLP classifier
    |
    +---> Part 4: Wrap-Up and Next Steps
    |
    +---> Appendix I: Installation guide
    +---> Appendix II: References and additional resources
    |
    v
Work through the homework notebook
    |
    +-- Section 1: Tensor creation, reshape, broadcasting
    +-- Section 2: Autograd on a multi-variable function
    +-- Section 3: MLP trained on California housing dataset
```

---

## Step-by-Step Walkthrough

### Part 0 - About and Motivation

PyTorch builds its computation graph dynamically as Python executes - meaning you can use ordinary Python control flow (`if`, `for`, `while`) and inspect intermediate tensor values at any point during a forward pass. This "define-by-run" behavior is why PyTorch became dominant in research: broken models are far easier to debug when you can print tensors mid-execution. The notebook opens with a side-by-side comparison of PyTorch and TensorFlow 2.x to orient learners who may have encountered one framework already.

### Part 1.1 - PyTorch Setup

Version verification and device detection. The install command depends on your OS and whether you have a CUDA GPU, so the notebook links to the official install selector at pytorch.org rather than hardcoding a wheel that may be outdated. Detecting the available device early (`cuda` vs `cpu`) is a habit worth building - it lets you move data and models to the right device without rewriting code later.

### Part 1.2 - Tensor Creation

Tensors are created with factory functions (`torch.tensor`, `torch.zeros`, `torch.randn`, `torch.arange`) rather than class constructors. The `torch.from_numpy` / `.numpy()` bridge is covered alongside a key warning: these operations share memory on CPU, so mutating the NumPy array also mutates the tensor. Use `.clone()` when you need an independent copy. Knowing this prevents subtle bugs when preprocessing data in NumPy and then converting to PyTorch.

### Part 1.3 - Tensor Operations

Operations follow NumPy conventions with operator overloading for the common cases. The section explicitly distinguishes element-wise multiplication (`torch.mul` / `*`) from matrix multiplication (`torch.matmul` / `@`) because confusing the two is a common source of shape errors. Broadcasting rules are identical to NumPy. In-place operations use a trailing underscore convention (`add_`, `mul_`) and save memory but must be avoided on tensors that require gradients - in-place modifications corrupt the computation graph.

### Part 1.4 - Indexing, Slicing, and Reshaping

Integer indexing removes a dimension; slice indexing preserves it - this distinction matters when assembling batch dimensions. `.view()` returns a new shape over the same underlying storage (requires contiguous memory) and is slightly faster. `.reshape()` handles the non-contiguous case by copying if needed. Passing `-1` as a dimension lets PyTorch infer the correct size, which reduces the chance of introducing hardcoded batch sizes.

### Part 2.1 - requires_grad and the Computation Graph

Setting `requires_grad=True` on a tensor tells PyTorch to record every downstream operation in a chain of `grad_fn` nodes. This chain is the computation graph. Leaf tensors (created directly by the user) have no `grad_fn`; intermediate tensors (produced by operations) do. Understanding this distinction helps explain why you can only call `.backward()` on tensors that descend from a leaf with `requires_grad=True`.

### Part 2.2 - Calling .backward()

`.backward()` walks the `grad_fn` chain in reverse and accumulates the chain-rule product at each leaf's `.grad`. Gradients accumulate across calls - they are not cleared automatically. The training loop therefore must call `optimizer.zero_grad()` before each new backward pass, or gradients from previous steps will add to the current ones and corrupt the update. The section demonstrates both a single-variable example and a multi-variable function to build intuition before the full training loop.

### Part 2.3 - Stopping Gradient Tracking

`torch.no_grad()` disables graph construction for a block of code. PyTorch stores intermediate activations in the computation graph during the forward pass so it can replay them during `.backward()`. At inference time you don't need those activations, so `no_grad()` reduces peak memory and speeds up evaluation. `.detach()` is the per-tensor alternative: it creates a tensor sharing the same storage but excluded from any graph.

### Part 3.1 - Building a Model with nn.Module

Every custom model subclasses `nn.Module` and implements two methods: `__init__` (define layers) and `forward` (define the computation). PyTorch tracks all parameters registered as submodule attributes, making them available via `model.parameters()` for the optimizer. The section also shows `nn.Sequential` as a concise shorthand for simple feedforward stacks. You call the model instance like a function - PyTorch internally calls `forward` and also runs any registered hooks.

### Part 3.2 - The Training Loop

The four-step pattern per batch: `optimizer.zero_grad()` - forward pass - `loss.backward()` - `optimizer.step()`. The example fits a linear model (`nn.Linear`) to noisy data drawn from the known function $y = 2x + 1$, using MSE loss and SGD. The choice of a known function lets you verify the learned weights (W close to 2, b close to 1) and confirm the loop is working before applying it to a real dataset.

### Part 3.3 - Evaluating the Model

`model.eval()` and `torch.no_grad()` serve different purposes and both are needed at inference time. `eval()` flips behavior flags in layers like Dropout (which would randomly zero activations during training but should not during evaluation) and BatchNorm (which should use running statistics rather than batch statistics). `no_grad()` stops PyTorch from building the computation graph. Omitting either one is a common bug that gives subtly wrong results.

### Homework - Generalization to Real Data

The homework transfers all three topics to a real regression problem: the California housing dataset (20,640 samples, 8 features). This forces the learner to handle feature standardization with scikit-learn's `StandardScaler`, convert NumPy arrays to `float32` tensors, define a multi-layer MLP (`nn.Module` with two hidden layers), and run the full training loop with the Adam optimizer. The progression from the toy problem to real data is deliberate - it confirms that the patterns learned in the main notebook generalize without modification.

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

# 4. Launch Jupyter and open the main notebook
jupyter notebook pytorch_introduction.ipynb
```

Run all cells top-to-bottom (Kernel - Restart and Run All). After completing the main notebook, open the homework:

```bash
jupyter notebook assets/homeworks/pytorch_introduction_homework.ipynb
```

The homework requires `scikit-learn` (included in `requirements.txt`) for the California housing dataset. No GPU is required - all computations run on CPU.

---

## Key Concepts Glossary

| Term | Plain-language definition |
|---|---|
| Tensor | A multi-dimensional array - the basic unit of data in PyTorch, analogous to a NumPy ndarray |
| Rank | The number of dimensions of a tensor (scalar = rank 0, vector = rank 1, matrix = rank 2) |
| `torch.tensor` | Factory function (lowercase `t`) that copies data into a new tensor |
| `requires_grad` | Flag that tells PyTorch to track operations on a tensor for gradient computation |
| `grad_fn` | The function node recorded on a tensor that links it back to the operation that created it |
| `.backward()` | Computes gradients of a scalar output with respect to all leaf tensors in the graph using the chain rule |
| `.grad` | Attribute on a leaf tensor that holds accumulated gradients after `.backward()` |
| `autograd` | PyTorch's automatic differentiation engine - derives gradients from the forward computation |
| `nn.Module` | Base class for all neural network models in PyTorch; provides parameter tracking and device management |
| `forward` | Method on an `nn.Module` that defines the computation for a forward pass |
| `optimizer.zero_grad()` | Clears accumulated gradients before a new backward pass - must be called each training step |
| `optimizer.step()` | Updates model parameters in the direction that reduces the loss, using the computed gradients |
| `model.eval()` | Switches layer behavior for inference - disables Dropout randomness and uses BatchNorm running stats |
| `torch.no_grad()` | Context manager that disables gradient graph construction, saving memory and compute during inference |
| Broadcasting | Automatically expanding a smaller tensor to match the shape of a larger one during operations, following NumPy rules |

---

## Further Reading

- PyTorch Official Tutorials
- Deep Learning with PyTorch
- Programming PyTorch for Deep Learning
- PyTorch Documentation - torch.autograd
- PyTorch Documentation - torch.nn

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
