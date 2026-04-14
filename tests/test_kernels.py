"""Kernel tests against GPJax reference implementation."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from gpjax.kernels.stationary import (
    RBF as GPJaxRBF,
    Matern52 as GPJaxMatern52,
    Matern32 as GPJaxMatern32,
)

from ptgp.kernels import ExpQuad, Matern52, Matern32, Matern12, RandomWalk

# GPJax uses float32 internally, so comparisons are limited to ~1e-6 precision.
ATOL = 1e-5


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


def _gpjax_gram(kernel, X_np):
    X_jnp = jnp.array(X_np, dtype=jnp.float32)
    return np.array(kernel.gram(X_jnp).to_dense())


def _gpjax_cross(kernel, X_np, Y_np):
    return np.array(kernel.cross_covariance(
        jnp.array(X_np, dtype=jnp.float32),
        jnp.array(Y_np, dtype=jnp.float32),
    ))


@pytest.fixture
def X_1d():
    return np.linspace(0.0, 5.0, 20)[:, None].astype(np.float64)


@pytest.fixture
def X_1d_other():
    return np.linspace(1.0, 3.0, 10)[:, None].astype(np.float64)


@pytest.fixture
def X_2d():
    rng = np.random.default_rng(42)
    return rng.standard_normal((15, 2)).astype(np.float64)


@pytest.fixture
def X_2d_other():
    rng = np.random.default_rng(99)
    return rng.standard_normal((8, 2)).astype(np.float64)


class TestExpQuad:
    def test_gram_1d(self, X_1d):
        ls, eta = 1.5, 2.0
        ptgp_k = eta**2 * ExpQuad(ls=ls)
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.5, 2.0
        ptgp_k = eta**2 * ExpQuad(ls=ls)
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d, X_1d_other), _gpjax_cross(gpjax_k, X_1d, X_1d_other), atol=ATOL)

    def test_gram_2d(self, X_2d):
        ptgp_k = ExpQuad(ls=0.8)
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.0))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), _gpjax_gram(gpjax_k, X_2d), atol=ATOL)

    def test_symmetric_annotation(self, X_1d):
        X_pt = pt.as_tensor_variable(X_1d)
        K = ExpQuad(ls=1.0)(X_pt)
        from pytensor.tensor.assumptions.specify import SpecifyAssumptions
        assert isinstance(K.owner.op, SpecifyAssumptions)
        assert "symmetric" in K.owner.op.assumptions
        assert "positive_definite" in K.owner.op.assumptions

    def test_cross_no_annotation(self, X_1d, X_1d_other):
        X_pt = pt.as_tensor_variable(X_1d)
        Y_pt = pt.as_tensor_variable(X_1d_other)
        K = ExpQuad(ls=1.0)(X_pt, Y_pt)
        from pytensor.tensor.assumptions.specify import SpecifyAssumptions
        assert not isinstance(K.owner.op, SpecifyAssumptions)


class TestMatern52:
    def test_gram_1d(self, X_1d):
        ls, eta = 1.2, 1.5
        ptgp_k = eta**2 * Matern52(ls=ls)
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.2, 1.5
        ptgp_k = eta**2 * Matern52(ls=ls)
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d, X_1d_other), _gpjax_cross(gpjax_k, X_1d, X_1d_other), atol=ATOL)

    def test_gram_2d(self, X_2d):
        ls, eta = 0.5, 2.0
        ptgp_k = eta**2 * Matern52(ls=ls)
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), _gpjax_gram(gpjax_k, X_2d), atol=ATOL)


class TestMatern32:
    def test_gram_1d(self, X_1d):
        ptgp_k = Matern32(ls=2.0)
        gpjax_k = GPJaxMatern32(lengthscale=jnp.array(2.0), variance=jnp.array(1.0))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_2d(self, X_2d, X_2d_other):
        ls, eta = 0.7, 1.3
        ptgp_k = eta**2 * Matern32(ls=ls)
        gpjax_k = GPJaxMatern32(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d, X_2d_other), _gpjax_cross(gpjax_k, X_2d, X_2d_other), atol=ATOL)


class TestMatern12:
    def test_gram_symmetry(self, X_1d):
        K = _ptgp_eval(Matern12(ls=1.0), X_1d)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_diagonal_is_one(self, X_1d):
        K = _ptgp_eval(Matern12(ls=1.0), X_1d)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-14)

    def test_positive_definite(self, X_1d):
        eigvals = np.linalg.eigvalsh(_ptgp_eval(Matern12(ls=1.0), X_1d))
        assert np.all(eigvals > -1e-10)

    def test_cross_shape(self, X_1d, X_1d_other):
        assert _ptgp_eval(Matern12(ls=1.0), X_1d, X_1d_other).shape == (20, 10)


class TestCombination:
    def test_sum_kernel(self, X_1d):
        k1, k2 = ExpQuad(ls=1.0), 0.25 * Matern52(ls=2.0)
        K_sum = _ptgp_eval(k1 + k2, X_1d)
        np.testing.assert_allclose(K_sum, _ptgp_eval(k1, X_1d) + _ptgp_eval(k2, X_1d), atol=1e-14)

    def test_product_kernel(self, X_1d):
        k1, k2 = ExpQuad(ls=1.0), Matern52(ls=2.0)
        K_prod = _ptgp_eval(k1 * k2, X_1d)
        np.testing.assert_allclose(K_prod, _ptgp_eval(k1, X_1d) * _ptgp_eval(k2, X_1d), atol=1e-14)

    def test_scalar_multiply(self, X_1d):
        k = ExpQuad(ls=1.0)
        np.testing.assert_allclose(_ptgp_eval(3.0 * k, X_1d), 3.0 * _ptgp_eval(k, X_1d), atol=1e-14)

    def test_sum_cross_covariance(self, X_1d, X_1d_other):
        k1, k2 = ExpQuad(ls=1.0), Matern32(ls=1.5)
        K_sum = _ptgp_eval(k1 + k2, X_1d, X_1d_other)
        np.testing.assert_allclose(K_sum, _ptgp_eval(k1, X_1d, X_1d_other) + _ptgp_eval(k2, X_1d, X_1d_other), atol=1e-14)


class TestActiveDims:
    def test_active_dims_selects_columns(self, X_2d):
        k_2d = ExpQuad(ls=1.0, active_dims=[0])
        k_1d = ExpQuad(ls=1.0)
        np.testing.assert_allclose(_ptgp_eval(k_2d, X_2d), _ptgp_eval(k_1d, X_2d[:, :1]), atol=1e-14)


class TestRandomWalk:
    def test_gram_values(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(_ptgp_eval(RandomWalk(), X), np.minimum(X, X.T), atol=1e-14)

    def test_scaling(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(_ptgp_eval(4.0 * RandomWalk(), X), 4.0 * np.minimum(X, X.T), atol=1e-14)

    def test_cross(self):
        X = np.array([[1.0], [3.0]])
        Y = np.array([[2.0], [4.0]])
        K = _ptgp_eval(RandomWalk(), X, Y)
        np.testing.assert_allclose(K, np.minimum(X, Y.T), atol=1e-14)

    def test_positive_definite(self):
        X = np.linspace(0.1, 5.0, 20)[:, None]
        eigvals = np.linalg.eigvalsh(_ptgp_eval(RandomWalk(), X))
        assert np.all(eigvals > -1e-10)
