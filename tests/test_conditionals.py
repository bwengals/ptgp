"""Tests for base_conditional in ptgp.conditionals."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.conditionals import base_conditional
from ptgp.kernels import ExpQuad


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestBaseConditional:
    def test_prior_conditional_no_q(self):
        """Without q_sqrt, should return prior conditional p(f*|u=f)."""
        M, N = 5, 10
        rng = np.random.default_rng(0)
        ls = 1.0
        kernel = ExpQuad(ls=ls)

        Z = rng.uniform(0, 5, (M, 1))
        X = rng.uniform(0, 5, (N, 1))
        f = rng.standard_normal(M)

        Kmm = kernel(pt.as_tensor_variable(Z))
        Kmn = kernel(pt.as_tensor_variable(Z), pt.as_tensor_variable(X))
        Knn_diag = pt.diag(kernel(pt.as_tensor_variable(X)))

        fmean, fvar = base_conditional(Kmn, Kmm, Knn_diag, pt.as_tensor_variable(f))
        fm, fv = _eval(fmean, fvar)

        assert fm.shape == (N,)
        assert fv.shape == (N,)
        assert np.all(fv >= -1e-6)  # variance should be non-negative

    def test_whitened_adds_variance(self):
        """Adding q_sqrt should increase variance compared to delta posterior."""
        M, N = 4, 8
        rng = np.random.default_rng(1)
        kernel = ExpQuad(ls=1.0)

        Z = rng.uniform(0, 5, (M, 1))
        X = rng.uniform(0, 5, (N, 1))
        f = np.zeros(M)
        q_sqrt = np.eye(M) * 0.5

        Kmm = kernel(pt.as_tensor_variable(Z))
        Kmn = kernel(pt.as_tensor_variable(Z), pt.as_tensor_variable(X))
        Knn_diag = pt.diag(kernel(pt.as_tensor_variable(X)))

        _, fvar_delta = _eval(
            *base_conditional(Kmn, Kmm, Knn_diag, pt.as_tensor_variable(f), white=True)
        )
        _, fvar_q = _eval(
            *base_conditional(
                Kmn,
                Kmm,
                Knn_diag,
                pt.as_tensor_variable(f),
                q_sqrt=pt.as_tensor_variable(q_sqrt),
                white=True,
            )
        )

        assert np.all(fvar_q >= fvar_delta - 1e-10)
