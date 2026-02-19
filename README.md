# Assignment 02: Try a Neural Network (PyTorch)

**Related to**: Week 2–3 — Neural Networks with PyTorch  
**Estimated time**: ~1 week  

## Goal

Build and train simple neural networks on CIFAR‑10 in PyTorch, and develop intuition for:

- **Parameters** (learned by training) vs **hyperparameters** (chosen by you)
- The standard training loop: forward → loss → backward → optimizer step
- How model capacity + hyperparameters affect loss/accuracy and training dynamics

## What you will submit

- **One notebook** (`.ipynb`) or **one script** (`.py`) that runs end‑to‑end
- A short **write‑up section** in the notebook (markdown) answering the questions below
- Plots + tables required in each part (screenshots are fine if using a script)

## Rules / constraints (keep it manageable)

- You may train on **CPU or GPU**
- Keep runtime reasonable: aim for **≤ 20 epochs** for your main experiments
- Use a **validation split** (don’t only report training accuracy)
- Make runs reproducible: set a **random seed** (e.g., `torch.manual_seed(0)`) and log it
- Save your **best validation model** (checkpoint) and report which epoch was best

---

## Part A — Warm‑up: tensors, shapes, autograd (PyTorch basics)

Do these in a small “warm‑up” section before CIFAR‑10.

### A1. Tensor operations and shapes

- Create tensors with shapes `(3,)`, `(3, 4)`, `(4, 3)` and demonstrate:
  - Matrix multiply (`@`), element‑wise multiply, broadcasting
  - `view`/`reshape`, `permute`, `flatten`
- **Deliverable**: print each operation’s input/output shapes (no silent shape changes).

### A2. Autograd sanity check (intuition from Assignment 01)

Let y = (w x + b)^2 with scalar `x`, `w`, `b`.

- Compute gradients **two ways**:
  - by hand (show the derivative in markdown)
  - with PyTorch autograd (`backward()`)
- **Deliverable**: show that the gradients match numerically (within a tiny tolerance).

---

## Part B — Load + explore CIFAR‑10 (data pipeline)

### B1. Load the dataset

Use `torchvision.datasets.CIFAR10` and `DataLoader`.

- **Deliverable**: print dataset sizes, class names, and one batch shape.

### B2. Explore and visualize

- Inspect a single sample:
  - Print image shape and label
  - Display the image
- Visualize **a 5×2 grid** of random training images with labels
- Plot **class counts** (histogram / bar chart)
- **Deliverable**: figures + short interpretation (1–3 sentences)

### B3. Preprocessing (do it correctly)

Use transforms that make sense for CIFAR‑10:

- Convert to tensor in [0,1]
- Normalize by **per‑channel mean/std** (either compute from the training set or use the standard CIFAR‑10 mean/std)
- **Deliverable**: report the mean/std you used and explain why normalization helps optimization.

---

## Part C — Build a baseline model (MLP) + a correct training loop

### C1. Baseline model: simple MLP

Implement an MLP that flattens input `3×32×32 → 3072` and uses ReLU hidden layers.

- Example baseline: `3072 → 256 → 128 → 10`
- Use `nn.Module` and `nn.Linear`

Important:

- If you use `nn.CrossEntropyLoss`, **do not put Softmax in the model** (the loss expects raw logits).

### C2. Training loop requirements

Your loop must include:

- `model.train()` and `model.eval()`
- `torch.no_grad()` during evaluation
- Clearing gradients each step (e.g., `optimizer.zero_grad()` before `loss.backward()`)
- Accuracy computation (top‑1)
- Tracking and plotting:
  - train loss vs epoch
  - train accuracy vs epoch
  - val accuracy vs epoch
- **Deliverable**: the 3 plots + final train/val accuracy numbers.

### C3. Parameters vs hyperparameters (must demonstrate)

Add a section called **“Parameters vs Hyperparameters”**:

- **Parameters (learned)**:
  - Print each parameter tensor name + shape
  - Print total parameter count: `sum(p.numel() for p in model.parameters())`
- **Hyperparameters (chosen)**:
  - Log at least: learning rate, batch size, optimizer, number of epochs, hidden sizes
- **Deliverable**: a small table listing your chosen hyperparameters for the baseline run.

---

## Part D — Experiments (hyperparameters + model capacity)

Run **at least 6 total training runs** (including the baseline) and compare them fairly.

### D1. Hyperparameter experiments (pick at least 4)

Change **one thing at a time** relative to baseline (keep others fixed), for example:

- learning rate (e.g., `1e-3` vs `3e-4` vs `3e-3`)
- batch size (e.g., 64 vs 128 vs 256)
- optimizer (SGD+momentum vs Adam)
- weight decay (L2 regularization)
- dropout on hidden layers

### D2. Model capacity experiments (pick at least 2)

- Wider MLP (more hidden units)
- Deeper MLP (more layers)

### D3. Comparison deliverables

- A table with each run:
  - model summary (depth/width), optimizer, LR, batch size, weight decay, dropout
  - best val accuracy, final val accuracy, training time (rough)
- One short paragraph answering:
  - Which hyperparameter mattered most for you and why?
  - Evidence of underfitting vs overfitting (use your curves)

---

## Common pitfalls (read before you debug for hours)

- If using `nn.CrossEntropyLoss`, **don’t add Softmax** in the model (logits only).
- If you forget `model.eval()` and `torch.no_grad()` during validation, your validation metrics can be misleading and slower.
- If training is “stuck”:
  - check input normalization
  - try a different learning rate
  - verify labels are integers in `[0..9]` and loss decreases on a tiny subset (overfit 200 samples test)

## Resources

- PyTorch “60‑Minute Blitz”: `https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html`
- `torchvision` CIFAR‑10: `https://pytorch.org/vision/stable/datasets.html#cifar`
- `nn.CrossEntropyLoss`: `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
- Matplotlib: `https://matplotlib.org/stable/contents.html`
