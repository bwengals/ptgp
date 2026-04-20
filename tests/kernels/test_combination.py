"""Kernel combination (sum, product, scalar multiply) tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad, Matern32, Matern52


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


@pytest.fixture
def X_1d():
    return np.linspace(0.0, 5.0, 20)[:, None].astype(np.float64)


@pytest.fixture
def X_1d_other():
    return np.linspace(1.0, 3.0, 10)[:, None].astype(np.float64)


class TestCombination:
    def test_sum_kernel(self, X_1d):
        k1, k2 = ExpQuad(input_dim=1, ls=1.0), 0.25 * Matern52(input_dim=1, ls=2.0)
        K_sum = _ptgp_eval(k1 + k2, X_1d)
        np.testing.assert_allclose(K_sum, _ptgp_eval(k1, X_1d) + _ptgp_eval(k2, X_1d), atol=1e-14)

    def test_product_kernel(self, X_1d):
        k1, k2 = ExpQuad(input_dim=1, ls=1.0), Matern52(input_dim=1, ls=2.0)
        K_prod = _ptgp_eval(k1 * k2, X_1d)
        np.testing.assert_allclose(K_prod, _ptgp_eval(k1, X_1d) * _ptgp_eval(k2, X_1d), atol=1e-14)

    def test_scalar_multiply(self, X_1d):
        k = ExpQuad(input_dim=1, ls=1.0)
        np.testing.assert_allclose(_ptgp_eval(3.0 * k, X_1d), 3.0 * _ptgp_eval(k, X_1d), atol=1e-14)

    def test_sum_cross_covariance(self, X_1d, X_1d_other):
        k1, k2 = ExpQuad(input_dim=1, ls=1.0), Matern32(input_dim=1, ls=1.5)
        K_sum = _ptgp_eval(k1 + k2, X_1d, X_1d_other)
        np.testing.assert_allclose(
            K_sum, _ptgp_eval(k1, X_1d, X_1d_other) + _ptgp_eval(k2, X_1d, X_1d_other), atol=1e-14
        )

    def test_input_dim_mismatch(self):
        k1 = ExpQuad(input_dim=2, ls=1.0)
        k2 = Matern52(input_dim=3, ls=1.0)
        with pytest.raises(ValueError, match="input_dim"):
            _ = k1 + k2
        with pytest.raises(ValueError, match="input_dim"):
            _ = k1 * k2
