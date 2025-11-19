import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Training data
# -----------------------------
# -----------------------------
# function f_1

def f_1(x, check_domain: bool = False):
    """Compute f_1(x) = e^{-2x} cos(5π x) + x for x in [0,1].

    Supports scalar (int/float), list/tuple, numpy.ndarray, and torch.Tensor.
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
# Training utilities (exposed for tests)
# -----------------------------

def build_model(hidden: int = 50) -> nn.Module:
    """Builds and returns a small MLP or single linear layer when hidden==0."""
    if hidden > 0:
        return nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    else:
        return nn.Linear(1, 1)


def train_model(n_train: int = 200,
                hidden: int = 50,
                epochs: int = 2000,
                lr: float = 1e-3,
                eval_n: int = 200,
                verbose: bool = True):
    """Train a model to approximate f_1 and return results.

    Returns: model, mse_eval, xs_eval, y_true_eval, y_pred_eval
    """
    x_train = torch.linspace(0.0, 1.0, steps=n_train).unsqueeze(1)
    y_train = f_1(x_train)

    model = build_model(hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"Epoch {epoch}/{epochs} — loss: {loss.item():.6f}")

    # Evaluation on a grid
    xs_eval = np.linspace(0.0, 1.0, eval_n)
    y_true_eval = f_1(xs_eval)
    x_eval_t = torch.from_numpy(xs_eval.astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        y_pred_eval = model(x_eval_t).squeeze().cpu().numpy()

    mse_eval = float(np.mean((y_true_eval - y_pred_eval) ** 2))
    if verbose:
        print(f"Evaluated on {eval_n} points — MSE: {mse_eval:.6f}")

    return model, mse_eval, xs_eval, y_true_eval, y_pred_eval


def plot_comparison(xs, y_true, y_pred, out_fname='f1_fit.png'):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, y_true, label='true f_1', lw=2)
    plt.plot(xs, y_pred, label='model prediction', lw=2, linestyle='--')
    plt.title('True f_1 vs model approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_fname, dpi=150)
    print(f"Saved comparison plot to {out_fname}")
    # end of script


def train_polynomial_neuron(degree: int = 22,
                            n_train: int = 200,
                            epochs: int = 2000,
                            lr: float = 1e-3,
                            eval_n: int = 400,
                            verbose: bool = True):
    """Train a single linear neuron on polynomial features x, x^2, ..., x^degree.

    Returns: model, mse_eval, xs_eval, y_true_eval, y_pred_eval
    """
    # Prepare data
    xs = np.linspace(0.0, 1.0, n_train)
    # features: columns x^1, x^2, ..., x^degree
    X = np.vstack([xs ** k for k in range(1, degree + 1)]).T.astype(np.float32)
    X_t = torch.from_numpy(X)
    x_tensor = torch.from_numpy(xs.astype(np.float32)).unsqueeze(1)
    y_t = f_1(x_tensor)

    model = nn.Linear(degree, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = criterion(y_pred, y_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"Poly Epoch {epoch}/{epochs} — loss: {loss.item():.6f}")

    # evaluation
    xs_eval = np.linspace(0.0, 1.0, eval_n)
    X_eval = np.vstack([xs_eval ** k for k in range(1, degree + 1)]).T.astype(np.float32)
    X_eval_t = torch.from_numpy(X_eval)
    with torch.no_grad():
        y_pred_eval = model(X_eval_t).squeeze().cpu().numpy()
    y_true_eval = f_1(xs_eval)
    mse_eval = float(np.mean((y_true_eval - y_pred_eval) ** 2))
    if verbose:
        print(f"Poly evaluated on {eval_n} points — MSE: {mse_eval:.6f}")

    return model, mse_eval, xs_eval, y_true_eval, y_pred_eval


if __name__ == "__main__":
    # Train the single linear neuron on polynomial features (degree 22)
    poly_degree = 22
    model_poly, mse_poly, xs_eval, y_true_eval, y_pred_eval = train_polynomial_neuron(
        degree=poly_degree, n_train=300, epochs=2000, lr=1e-3, eval_n=400, verbose=True
    )
    print(f"Polynomial neuron (degree {poly_degree}) MSE: {mse_poly:.6f}")
    plot_comparison(xs_eval, y_true_eval, y_pred_eval, out_fname='f1_poly_fit.png')
