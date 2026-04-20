"""Non-stationary kernel tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad, Gibbs, Matern52, RandomWalk, WarpedInput


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


class TestRandomWalk:
    def test_gram_values(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(_ptgp_eval(RandomWalk(), X), np.minimum(X, X.T), atol=1e-14)

    def test_scaling(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(
            _ptgp_eval(4.0 * RandomWalk(), X), 4.0 * np.minimum(X, X.T), atol=1e-14
        )

    def test_cross(self):
        X = np.array([[1.0], [3.0]])
        Y = np.array([[2.0], [4.0]])
        K = _ptgp_eval(RandomWalk(), X, Y)
        np.testing.assert_allclose(K, np.minimum(X, Y.T), atol=1e-14)

    def test_positive_definite(self):
        X = np.linspace(0.1, 5.0, 20)[:, None]
        eigvals = np.linalg.eigvalsh(_ptgp_eval(RandomWalk(), X))
        assert np.all(eigvals > -1e-10)


class TestGibbs:
    def test_constant_ls_matches_expquad(self):
        """Gibbs with constant l(x) = l0 reduces to ExpQuad(ls=l0)."""
        X = np.linspace(0.0, 5.0, 15)[:, None]
        ls0 = 1.3
        gibbs = Gibbs(lengthscale_func=lambda X: pt.fill(X[:, 0], ls0))
        expquad = ExpQuad(input_dim=1, ls=ls0)
        np.testing.assert_allclose(_ptgp_eval(gibbs, X), _ptgp_eval(expquad, X), atol=1e-14)

    def test_vs_pymc(self):
        """Gibbs with varying l(x) matches PyMC's pm.gp.cov.Gibbs."""
        import pymc as pm

        X = np.linspace(0.0, 5.0, 15)[:, None]

        def ls_func(X):
            return 0.5 + 0.3 * X[:, 0]

        gibbs = Gibbs(lengthscale_func=ls_func)
        pymc_gibbs = pm.gp.cov.Gibbs(input_dim=1, lengthscale_func=ls_func)

        K_ptgp = _ptgp_eval(gibbs, X)
        K_pymc = pymc_gibbs(pt.as_tensor_variable(X)).eval()
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-10)

    def test_rejects_multidim_active_dims(self):
        with pytest.raises(ValueError, match="length 1"):
            Gibbs(lengthscale_func=lambda X: X[:, 0], active_dims=[0, 0])


class TestWarpedInput:
    def test_identity_warp_matches_inner(self):
        """Warping with the identity should give the same kernel as the inner kernel."""
        X = np.linspace(0.0, 5.0, 15)[:, None]
        inner = Matern52(input_dim=1, ls=0.8)
        warped = WarpedInput(input_dim=1, kernel_func=inner, warp_func=lambda X: X)
        np.testing.assert_allclose(_ptgp_eval(warped, X), _ptgp_eval(inner, X), atol=1e-14)

    def test_vs_pymc(self):
        """WarpedInput with a nonlinear warp matches PyMC's pm.gp.cov.WarpedInput."""
        import pymc as pm

        X = np.linspace(0.1, 3.0, 15)[:, None]

        def warp(X):
            return X**2

        inner = ExpQuad(input_dim=1, ls=1.0)
        pymc_inner = pm.gp.cov.ExpQuad(input_dim=1, ls=1.0)

        warped = WarpedInput(input_dim=1, kernel_func=inner, warp_func=warp)
        pymc_warped = pm.gp.cov.WarpedInput(input_dim=1, cov_func=pymc_inner, warp_func=warp)

        K_ptgp = _ptgp_eval(warped, X)
        K_pymc = pymc_warped(pt.as_tensor_variable(X)).eval()
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-10)
