from hover.utils.common_nn import BaseSequential, MLP, LogisticRegression
import numpy as np
import torch


def architecture_subroutine(architecture, dim_inp=300, dim_out=2, num_vecs=10):
    """
    Test a specific architecture.
    """
    nn = architecture(dim_inp, dim_out)
    inp = torch.Tensor(np.random.rand(num_vecs, dim_inp))
    out = nn(inp)
    assert out.shape == (num_vecs, dim_out)
    out = nn.eval_per_layer(inp)[-1]
    assert out.shape == (num_vecs, dim_out)


def test_MLP():
    architecture_subroutine(MLP)


def test_LR():
    architecture_subroutine(LogisticRegression)
