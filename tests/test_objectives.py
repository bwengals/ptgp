"""Tests for objectives, KL divergence, and conditionals."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.inducing_variables import InducingPoints
from ptgp.gp import GP
from ptgp.svgp import SVGP
from ptgp.vfe import VFE
from ptgp.kl import gauss_kl
from ptgp.conditionals import base_conditional
from ptgp.objectives import marginal_log_likelihood, elbo, collapsed_elbo


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 20))[:, None].astype(np.float64)
    y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(20)
    return X, y


@pytest.fixture
def inducing_points():
    return np.linspace(0.5, 4.5, 5)[:, None].astype(np.float64)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

class TestGaussKL:
    def test_whitened_zero_mean_identity_cov(self):
        """KL[N(0, I) || N(0, I)] = 0."""
        M = 3
        kl = _eval(gauss_kl(pt.zeros(M), pt.eye(M), K=None))
        np.testing.assert_allclose(kl, 0.0, atol=1e-10)

    def test_whitened_nonzero_mean(self):
        """KL should be positive for non-trivial q."""
        q_mu = pt.as_tensor_variable(np.array([1.0, 0.5, -0.3]))
        q_sqrt = pt.as_tensor_variable(np.eye(3) * 0.5)
        kl = _eval(gauss_kl(q_mu, q_sqrt, K=None))
        assert kl > 0.0

    def test_unwhitened_matches_whitened_with_identity_prior(self):
        """With K=I, unwhitened should match whitened."""
        q_mu = pt.as_tensor_variable(np.array([0.5, -0.5]))
        q_sqrt = pt.as_tensor_variable(np.array([[0.8, 0.0], [0.2, 0.6]]))
        kl_w = _eval(gauss_kl(q_mu, q_sqrt, K=None))
        kl_u = _eval(gauss_kl(q_mu, q_sqrt, K=pt.eye(2)))
        np.testing.assert_allclose(kl_w, kl_u, atol=1e-10)

    def test_unwhitened_positive(self):
        M = 3
        rng = np.random.default_rng(0)
        L = np.tril(rng.standard_normal((M, M)))
        K = L @ L.T + 0.1 * np.eye(M)
        q_mu = rng.standard_normal(M)
        q_sqrt = np.eye(M) * 0.5

        kl = _eval(gauss_kl(
            pt.as_tensor_variable(q_mu),
            pt.as_tensor_variable(q_sqrt),
            K=pt.as_tensor_variable(K),
        ))
        assert kl > 0.0


# ---------------------------------------------------------------------------
# Base conditional
# ---------------------------------------------------------------------------

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

        _, fvar_delta = _eval(*base_conditional(
            Kmn, Kmm, Knn_diag, pt.as_tensor_variable(f), white=True))
        _, fvar_q = _eval(*base_conditional(
            Kmn, Kmm, Knn_diag, pt.as_tensor_variable(f),
            q_sqrt=pt.as_tensor_variable(q_sqrt), white=True))

        assert np.all(fvar_q >= fvar_delta - 1e-10)


# ---------------------------------------------------------------------------
# Marginal log likelihood (exact GP)
# ---------------------------------------------------------------------------

class TestMarginalLogLikelihood:
    def test_finite(self, regression_data):
        X, y = regression_data
        gp = GP(kernel=ExpQuad(ls=1.0), mean=Zero(), likelihood=Gaussian(sigma=0.1))
        mll = _eval(marginal_log_likelihood(gp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))
        assert np.isfinite(mll)

    def test_better_fit_higher_mll(self, regression_data):
        """A kernel with reasonable params should have higher MLL than a bad one."""
        X, y = regression_data
        gp_good = GP(kernel=ExpQuad(ls=1.0), mean=Zero(), likelihood=Gaussian(sigma=0.1))
        gp_bad = GP(kernel=ExpQuad(ls=0.01), mean=Zero(), likelihood=Gaussian(sigma=10.0))

        mll_good = _eval(marginal_log_likelihood(gp_good,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))
        mll_bad = _eval(marginal_log_likelihood(gp_bad,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        assert mll_good > mll_bad


# ---------------------------------------------------------------------------
# SVGP ELBO
# ---------------------------------------------------------------------------

class TestELBO:
    def test_finite(self, regression_data, inducing_points):
        X, y = regression_data
        svgp = SVGP(
            kernel=ExpQuad(ls=1.0), mean=Zero(), likelihood=Gaussian(sigma=0.1),
            inducing_variable=InducingPoints(pt.as_tensor_variable(inducing_points)),
        )
        elbo_val = _eval(elbo(svgp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))
        assert np.isfinite(elbo_val)

    def test_unwhitened_finite(self, regression_data, inducing_points):
        X, y = regression_data
        svgp = SVGP(
            kernel=ExpQuad(ls=1.0), mean=Zero(), likelihood=Gaussian(sigma=0.1),
            inducing_variable=InducingPoints(pt.as_tensor_variable(inducing_points)),
            whiten=False,
        )
        elbo_val = _eval(elbo(svgp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))
        assert np.isfinite(elbo_val)

    def test_whitened_and_unwhitened_agree_at_prior(self, regression_data, inducing_points):
        """With q=prior (q_mu=0, q_sqrt=I for whitened; q_mu=0, q_sqrt=Luu for unwhitened),
        both parameterizations should give the same ELBO."""
        X, y = regression_data
        Z = pt.as_tensor_variable(inducing_points)
        kernel = ExpQuad(ls=1.0)

        # Whitened: q_mu=0, q_sqrt=I is the prior q(v)=N(0,I)
        svgp_w = SVGP(
            kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=0.1),
            inducing_variable=InducingPoints(Z), whiten=True,
        )
        elbo_w = _eval(elbo(svgp_w,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        # Unwhitened: q_mu=0, q_sqrt=Luu is the prior q(u)=N(0, Kuu)
        Kuu = _eval(kernel(Z))
        Luu = np.linalg.cholesky(Kuu)
        svgp_u = SVGP(
            kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=0.1),
            inducing_variable=InducingPoints(Z), whiten=False,
            q_mu=pt.zeros(5), q_sqrt=pt.as_tensor_variable(Luu),
        )
        elbo_u = _eval(elbo(svgp_u,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        np.testing.assert_allclose(elbo_w, elbo_u, atol=1e-6)

    def test_elbo_less_than_mll(self, regression_data, inducing_points):
        """ELBO should be a lower bound on the marginal log likelihood."""
        X, y = regression_data
        ls, sigma = 1.0, 0.1
        kernel = ExpQuad(ls=ls)

        gp = GP(kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=sigma))
        mll_val = _eval(marginal_log_likelihood(gp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        svgp = SVGP(
            kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=sigma),
            inducing_variable=InducingPoints(pt.as_tensor_variable(inducing_points)),
        )
        elbo_val = _eval(elbo(svgp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        assert elbo_val <= mll_val + 1e-6  # ELBO <= MLL


# ---------------------------------------------------------------------------
# Collapsed ELBO (VFE)
# ---------------------------------------------------------------------------

class TestCollapsedELBO:
    def test_finite(self, regression_data, inducing_points):
        X, y = regression_data
        vfe_model = VFE(
            kernel=ExpQuad(ls=1.0), mean=Zero(), likelihood=Gaussian(sigma=0.1),
            inducing_variable=InducingPoints(pt.as_tensor_variable(inducing_points)),
        )
        celbo = _eval(collapsed_elbo(vfe_model,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))
        assert np.isfinite(celbo)

    def test_collapsed_elbo_less_than_mll(self, regression_data, inducing_points):
        """Collapsed ELBO should be a lower bound on the marginal log likelihood."""
        X, y = regression_data
        ls, sigma = 1.0, 0.1
        kernel = ExpQuad(ls=ls)

        gp = GP(kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=sigma))
        mll_val = _eval(marginal_log_likelihood(gp,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        vfe_model = VFE(
            kernel=kernel, mean=Zero(), likelihood=Gaussian(sigma=sigma),
            inducing_variable=InducingPoints(pt.as_tensor_variable(inducing_points)),
        )
        celbo = _eval(collapsed_elbo(vfe_model,
            pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        assert celbo <= mll_val + 1e-6  # collapsed ELBO <= MLL
