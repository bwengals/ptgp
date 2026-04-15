"""Non-stationary kernel tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.kernels import RandomWalk


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
