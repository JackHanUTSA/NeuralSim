import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

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
        return nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    else:
        return nn.Linear(1, 1)


def train_model(n_train: int = 2000,
                hidden: int = 1,
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


def plot_comparison(xs, y_true, y_pred, out_fname='f1_fit.svg', formula: str = None):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, y_true, label='true f_1', lw=2)
    plt.plot(xs, y_pred, label='model prediction', lw=2, linestyle='--')
    plt.title('True f_1 vs model approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    # If a formula string is provided, render it on the plot (supports mathtext)
    if formula is not None:
        try:
            plt.text(0.02, 0.95, formula, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top')
        except Exception:
            # Fallback: plain text
            plt.text(0.02, 0.95, formula, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(out_fname)
    print(f"Saved comparison plot to {out_fname}")
    # end of script


def plot_metrics(metrics: dict, out_fname: str = 'f1_poly_metrics.svg'):
    """Create a separate chart showing MSE (train/val/test), rL2 (train/val/test), and training time."""
    labels = ['train', 'val', 'test']
    mses = [metrics['mse_train'], metrics['mse_val'], metrics['mse_test']]
    rls = [metrics['rL2_train'], metrics['rL2_val'], metrics['rL2_test']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # MSE bar
    axes[0].bar(labels, mses, color=['C0', 'C1', 'C2'])
    axes[0].set_title('MSE')
    axes[0].set_ylabel('Mean Squared Error')

    # rL2 bar
    axes[1].bar(labels, rls, color=['C0', 'C1', 'C2'])
    axes[1].set_title('rL2 (relative L2)')
    axes[1].set_ylabel('Relative L2 norm')

    # Training time and counts
    axes[2].axis('off')
    txt = (f"training_time: {metrics['training_time']:.3f}s\n"
        f"n_train: {metrics.get('n_train', '?')}, n_val: {metrics.get('n_val', '?')}, n_test: {metrics.get('n_test', '?')}")
    axes[2].text(0.1, 0.6, txt, fontsize=10)
    axes[2].set_title('Train info')

    plt.tight_layout()
    plt.savefig(out_fname)
    print(f"Saved metrics plot to {out_fname}")


def plot_learning_curves(metrics: dict, out_fname: str = 'f1_learning_curves.svg'):
    """Plot training and validation loss vs epoch from metrics dict and save figure."""
    train_losses = metrics.get('train_losses', None)
    val_losses = metrics.get('val_losses', None)
    if train_losses is None or val_losses is None:
        print("No learning curve data available in metrics.")
        return

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, val_losses, label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_fname)
    print(f"Saved learning curves to {out_fname}")


def train_polynomial_neuron(degree: int = 22,
                            n_points: int = 300,
                            epochs: int = 2000,
                            lr: float = 1e-3,
                            eval_n: int = 400,
                            val_frac: float = 0.1,
                            test_frac: float = 0.1,
                            seed: int = 42,
                            verbose: bool = True):
    """Train a single linear neuron on polynomial features x, x^2, ..., x^degree.

    Splits the sampled points into train/val/test according to val_frac/test_frac.
    Returns: model, metrics, xs_eval, y_true_eval, y_pred_eval

    metrics is a dict containing mse_train, mse_val, mse_test, rL2_train, rL2_val, rL2_test, training_time
    """
    # Prepare data
    rng = np.random.RandomState(seed)
    xs_all = np.linspace(0.0, 1.0, n_points)
    indices = np.arange(n_points)
    rng.shuffle(indices)

    n_test = int(np.floor(test_frac * n_points))
    n_val = int(np.floor(val_frac * n_points))
    n_train = n_points - n_val - n_test

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    xs_train = xs_all[idx_train]
    xs_val = xs_all[idx_val]
    xs_test = xs_all[idx_test]

    def poly_features(xs):
        return np.vstack([xs ** k for k in range(1, degree + 1)]).T.astype(np.float32)

    X_train = poly_features(xs_train)
    X_val = poly_features(xs_val)
    X_test = poly_features(xs_test)

    X_train_t = torch.from_numpy(X_train)
    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)

    y_train = f_1(torch.from_numpy(xs_train.astype(np.float32)).unsqueeze(1))
    y_val = f_1(torch.from_numpy(xs_val.astype(np.float32)).unsqueeze(1))
    y_test = f_1(torch.from_numpy(xs_test.astype(np.float32)).unsqueeze(1))

    model = nn.Linear(degree, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.perf_counter()
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # compute validation loss each epoch
        with torch.no_grad():
            y_val_pred_epoch = model(X_val_t)
            val_loss = criterion(y_val_pred_epoch, y_val)

        train_losses.append(float(loss.item()))
        val_losses.append(float(val_loss.item()))

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"Poly Epoch {epoch}/{epochs} — train_loss: {loss.item():.6f}, val_loss: {val_loss.item():.6f}")
    training_time = time.perf_counter() - start

    with torch.no_grad():
        y_train_pred = model(X_train_t).squeeze().cpu().numpy()
        y_val_pred = model(X_val_t).squeeze().cpu().numpy()
        y_test_pred = model(X_test_t).squeeze().cpu().numpy()

    y_train_true = y_train.squeeze().cpu().numpy()
    y_val_true = y_val.squeeze().cpu().numpy()
    y_test_true = y_test.squeeze().cpu().numpy()

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    def rL2(a, b):
        # relative L2 norm: ||a-b||_2 / ||a||_2
        num = np.linalg.norm(a - b)
        den = np.linalg.norm(a)
        return float(num / den) if den != 0 else float('inf')

    metrics = {
        'mse_train': mse(y_train_true, y_train_pred),
        'mse_val': mse(y_val_true, y_val_pred),
        'mse_test': mse(y_test_true, y_test_pred),
        'rL2_train': rL2(y_train_true, y_train_pred),
        'rL2_val': rL2(y_val_true, y_val_pred),
        'rL2_test': rL2(y_test_true, y_test_pred),
        'training_time': training_time,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
    }
    # include learning curves
    metrics['train_losses'] = train_losses
    metrics['val_losses'] = val_losses

    # evaluation on dense grid for plotting
    xs_eval = np.linspace(0.0, 1.0, eval_n)
    X_eval = poly_features(xs_eval)
    X_eval_t = torch.from_numpy(X_eval)
    with torch.no_grad():
        y_pred_eval = model(X_eval_t).squeeze().cpu().numpy()
    y_true_eval = f_1(xs_eval)

    if verbose:
        print(f"Poly evaluated on {eval_n} points — MSE (eval grid): {float(np.mean((y_true_eval - y_pred_eval) ** 2)):.6f}")

    return model, metrics, xs_eval, y_true_eval, y_pred_eval


if __name__ == "__main__":
    # Train the single linear neuron on polynomial features (degree 22)
    poly_degree = 22
    # Increase input sampling to 10000 points. To keep runtime reasonable we lower epochs to 500.
    # Increase sampled points to 50,000 as requested. Warning: full-batch training with
    # 50k samples and 2000 epochs can take a long time. Consider reducing epochs or
    # using closed-form least-squares for an exact solution.
    model_poly, metrics_poly, xs_eval, y_true_eval, y_pred_eval = train_polynomial_neuron(
        degree=poly_degree, n_points=50000, epochs=2000, lr=1e-3, eval_n=400, verbose=True, seed=42
    )
    print(f"Polynomial neuron (degree {poly_degree}) MSE (test): {metrics_poly['mse_test']:.6f}")
    formula_str = r"$f_1(x)=e^{-2x}\cos(5\pi x)+x$"
    plot_comparison(xs_eval, y_true_eval, y_pred_eval, out_fname='f1_poly_fit.svg', formula=formula_str)
    plot_metrics(metrics_poly, out_fname='f1_poly_metrics.svg')
    plot_learning_curves(metrics_poly, out_fname='f1_learning_curves.svg')
    # save metrics to disk for downstream tools
    try:
        import json
        metrics_out = {k: float(v) if isinstance(v, (np.floating, float, int)) else v for k, v in metrics_poly.items() if k not in ('train_losses', 'val_losses')}
        # ensure ints remain ints
        for k in ('n_train', 'n_val', 'n_test'):
            if k in metrics_poly:
                metrics_out[k] = int(metrics_poly[k])
        with open('f1_poly_metrics.json', 'w') as fh:
            json.dump(metrics_out, fh, indent=2)
        # also save losses as npz
        np.savez('f1_poly_metrics.npz', train_losses=np.array(metrics_poly.get('train_losses', [])), val_losses=np.array(metrics_poly.get('val_losses', [])), **{k: metrics_poly[k] for k in ['mse_train','mse_val','mse_test','rL2_train','rL2_val','rL2_test','training_time','n_train','n_val','n_test']})
        print('Saved metrics to f1_poly_metrics.json and f1_poly_metrics.npz')
    except Exception:
        pass
