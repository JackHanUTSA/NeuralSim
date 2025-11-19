import numpy as np
import pytest
import torch
import random

from main import train_model


def make_deterministic(seed: int = 12345):
    """Set seeds and deterministic flags for CI-friendly behavior."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # Try to enable deterministic algorithms where available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older torch versions may not support this; set backend flags as fallback
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_quick_training_mse_deterministic():
    # Make the test deterministic to be CI-friendly
    make_deterministic(seed=42)

    # Increase train size / hidden units and epochs a bit to get a tighter MSE
    # while keeping runtime reasonable for CI.
    model, mse, xs, y_true, y_pred = train_model(n_train=100, hidden=30, epochs=600, lr=1e-3, eval_n=200, verbose=False)
    assert isinstance(mse, float)
    # Stricter MSE threshold — tuned to pass reliably under deterministic conditions
    assert mse < 0.13
