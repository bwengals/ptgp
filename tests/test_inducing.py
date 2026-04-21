"""Tests for inducing point initialization strategies."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.inducing import (
    Points,
    greedy_variance_init,
    kmeans_init,
    random_subsample_init,
)
from ptgp.kernels import ExpQuad


def _eval_kernel(kernel, X, Y=None):
    X_pt = pt.as_tensor_variable(X)
    Y_pt = pt.as_tensor_variable(Y) if Y is not None else None
    return pytensor.function([], kernel(X_pt, Y_pt))()


def _nystrom_residual_trace(X, Z, kernel, jitter=1e-9):
    """tr(K(X,X) - K(X,Z) K(Z,Z)^{-1} K(Z,X))."""
    Kxx = _eval_kernel(kernel, X)
    Kxz = _eval_kernel(kernel, X, Z)
    Kzz = _eval_kernel(kernel, Z) + jitter * np.eye(len(Z))
    Q = Kxz @ np.linalg.solve(Kzz, Kxz.T)
    return float(np.trace(Kxx - Q))


@pytest.fixture
def gaussian_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]])
    X = np.vstack([c + 0.3 * rng.standard_normal((50, 2)) for c in centers])
    return X, centers


class TestRandomSubsample:
    def test_returns_inducing_points(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip = random_subsample_init(X, 10, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (10, 2)

    def test_rows_are_in_x(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip = random_subsample_init(X, 10, rng=0)
        for z in ip.Z:
            assert np.any(np.all(X == z, axis=1))

    def test_no_duplicates(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip = random_subsample_init(X, 30, rng=0)
        assert len(np.unique(ip.Z, axis=0)) == 30

    def test_reproducible(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip1 = random_subsample_init(X, 10, rng=42)
        ip2 = random_subsample_init(X, 10, rng=42)
        np.testing.assert_array_equal(ip1.Z, ip2.Z)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        with pytest.raises(ValueError, match="exceeds"):
            random_subsample_init(X, 20)


class TestKmeans:
    def test_returns_inducing_points(self, gaussian_blobs):
        X, _ = gaussian_blobs
        ip = kmeans_init(X, 3, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (3, 2)

    def test_recovers_cluster_centers(self, gaussian_blobs):
        """On well-separated blobs, centroids should lie near the true centers."""
        X, centers = gaussian_blobs
        ip = kmeans_init(X, 3, rng=0)
        dists = np.linalg.norm(ip.Z[:, None, :] - centers[None, :, :], axis=-1)
        assert np.all(dists.min(axis=0) < 0.5)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(ValueError, match="exceeds"):
            kmeans_init(X, 20)


class TestGreedyVariance:
    def test_returns_inducing_points(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip = greedy_variance_init(X, 10, kernel, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (10, 2)

    def test_rows_are_in_x(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip = greedy_variance_init(X, 15, kernel, rng=0)
        for z in ip.Z:
            assert np.any(np.all(np.isclose(X, z), axis=1))

    def test_beats_random_on_average(self):
        """Greedy space-filling selection yields lower Nystrom residual than random."""
        X = np.linspace(0.0, 10.0, 100)[:, None]
        kernel = ExpQuad(input_dim=1, ls=1.0)
        M = 10

        greedy_err = _nystrom_residual_trace(X, greedy_variance_init(X, M, kernel, rng=0).Z, kernel)
        random_errs = [
            _nystrom_residual_trace(X, random_subsample_init(X, M, rng=s).Z, kernel) for s in range(10)
        ]
        assert greedy_err < np.mean(random_errs)

    def test_threshold_terminates_early(self):
        """With a large threshold, fewer than M points should be returned."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip = greedy_variance_init(X, 25, kernel, threshold=1e6, rng=0)
        assert ip.Z.shape[0] < 25

    def test_rejects_non_kernel(self):
        X = np.random.default_rng(0).standard_normal((20, 2))
        with pytest.raises(TypeError, match="ptgp Kernel"):
            greedy_variance_init(X, 5, kernel=lambda X, Y=None: X @ X.T)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        with pytest.raises(ValueError, match="exceeds"):
            greedy_variance_init(X, 20, kernel)


class TestIntegrationWithSVGP:
    """Sanity check: output feeds into SVGP without hand-wrapping in pt.as_tensor_variable."""

    def test_numpy_z_flows_into_kernel(self):
        from ptgp.gp import SVGP
        from ptgp.likelihoods import Gaussian
        from ptgp.objectives import elbo

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 1))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(40)
        kernel = ExpQuad(input_dim=1, ls=1.0)

        ip = greedy_variance_init(X, 5, kernel, rng=0)
        svgp = SVGP(
            kernel=kernel,
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=ip,
        )
        val = pytensor.function(
            [], elbo(svgp, pt.as_tensor_variable(X), pt.as_tensor_variable(y))
        )()
        assert np.isfinite(val)