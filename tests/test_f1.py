import numpy as np
import torch
import pytest

from main import f_1


def test_f1_scalar_values():
    # Known values: f_1(0)=1, f_1(0.5)=0.5
    assert pytest.approx(1.0, rel=1e-7) == f_1(0.0)
    assert pytest.approx(0.5, rel=1e-6) == f_1(0.5)


def test_f1_array_and_sequence():
    xs = [0.0, 0.5, 1.0]
    xs_np = np.array(xs)
    out_seq = f_1(xs)
    out_np = f_1(xs_np)
    # both should be numpy arrays and equal
    assert isinstance(out_np, np.ndarray)
    assert np.allclose(out_seq, out_np)


def test_f1_torch_tensor():
    xt = torch.tensor([0.0, 0.5, 1.0])
    yt = f_1(xt)
    assert torch.is_tensor(yt)
    # check first and last
    assert pytest.approx(1.0, rel=1e-7) == yt[0].item()
    assert pytest.approx(0.5, rel=1e-6) == yt[1].item()
