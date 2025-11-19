import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Training data
# -----------------------------
# Example: y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])


def f_1(x, check_domain: bool = False):
    """Compute f_1(x) = e^{-2x} cos(5π x) + x for x in [0,1].

    Supports scalar (int/float), list/tuple, numpy.ndarray, and torch.Tensor.

    Args:
        x: input value(s). Can be scalar, sequence, numpy array, or torch tensor.
        check_domain: if True, raise ValueError when any input is outside [0,1].

    Returns:
        Value(s) of f_1 in the same container type as the input (torch.Tensor for torch inputs,
        numpy.ndarray for sequence/ndarray inputs, float for scalar inputs).
    """
    # Torch tensor path
    if torch.is_tensor(x):
        x_t = x
        if check_domain:
            if ((x_t < 0) | (x_t > 1)).any():
                raise ValueError("f_1 input outside domain [0,1]")
        return torch.exp(-2.0 * x_t) * torch.cos(5.0 * math.pi * x_t) + x_t

    # Numpy / sequence path
    if isinstance(x, (list, tuple, np.ndarray)):
        x_arr = np.asarray(x, dtype=float)
        if check_domain:
            if np.any((x_arr < 0) | (x_arr > 1)):
                raise ValueError("f_1 input outside domain [0,1]")
        return np.exp(-2.0 * x_arr) * np.cos(5.0 * np.pi * x_arr) + x_arr

    # Scalar path (int/float)
    try:
        x_f = float(x)
    except Exception:
        raise TypeError("Unsupported input type for f_1: {}".format(type(x)))
    if check_domain and not (0.0 <= x_f <= 1.0):
        raise ValueError("f_1 input outside domain [0,1]")
    return math.exp(-2.0 * x_f) * math.cos(5.0 * math.pi * x_f) + x_f

# -----------------------------
# 2. Single linear neuron
# -----------------------------
model = nn.Linear(1, 1)   # one input → one output (single neuron)

# -----------------------------
# 3. Loss + optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -----------------------------
# 4. Training loop
# -----------------------------
for epoch in range(500):
    optimizer.zero_grad()        # reset gradients
    y_pred = model(X)            # forward pass
    loss = criterion(y_pred, Y)  # compute loss
    loss.backward()              # compute gradients
    optimizer.step()             # update weights

# -----------------------------
# 5. Results
# -----------------------------
w, b = model.parameters()
print("Learned weight:", w.item())
print("Learned bias:", b.item())

# Test the model
x_test = torch.tensor([[5.0]])
print("Input: 5 → Prediction:", model(x_test).item())

if __name__ == "__main__":
    # Quick self-test / demonstration of f_1
    print("\nTesting f_1: f_1(x) = e^{-2x} cos(5π x) + x on [0,1]")
    print("f_1(0) ->", f_1(0.0))
    print("f_1(1) ->", f_1(1.0))

    xs = np.linspace(0.0, 1.0, 5)
    print("f_1 on numpy linspace 5 points ->", f_1(xs))

    # Torch tensor example
    xt = torch.linspace(0.0, 1.0, steps=5)
    print("f_1 on torch tensor 5 points ->", f_1(xt))

    # Plot f_1 on a dense grid and save figure
    x_plot = np.linspace(0.0, 1.0, 400)
    y_plot = f_1(x_plot)
    plt.figure(figsize=(6, 4))
    plt.plot(x_plot, y_plot, label=r'$f_1(x)=e^{-2x}\cos(5\pi x)+x$')
    plt.title('Plot of $f_1$ on [0,1]')
    plt.xlabel('x')
    plt.ylabel('f_1(x)')
    plt.grid(True)
    plt.legend()
    out_fname = 'f1_plot.png'
    plt.tight_layout()
    plt.savefig(out_fname, dpi=150)
    print(f"Saved plot to {out_fname}")
