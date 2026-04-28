"""Inducing variables and initialization strategies.

The `*_init` functions return an :class:`Points` wrapping a plain
numpy array, so ``ip.Z`` is directly usable for plotting.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.cluster.vq

from ptgp.kernels.base import Kernel


class InducingVariables:
    """Base class for inducing variables.

    Enables dispatch to different implementations for inter-domain,
    multiscale, or structured inducing points.
    """

    @property
    def num_inducing(self):
        raise NotImplementedError


class Points(InducingVariables):
    """Standard real-space inducing points.

    Parameters
    ----------
    Z : tensor or PyMC random variable, shape (M, D)
        Inducing point locations.
    """

    def __init__(self, Z):
        self.Z = Z

    @property
    def num_inducing(self):
        return self.Z.shape[0]


def random_subsample_init(X, M, rng=None):
    """Select ``M`` inducing points uniformly at random from ``X``.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Candidate locations.
    M : int
        Number of inducing points.
    rng : int or numpy Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    Points
        Wrapping an ``(M, D)`` numpy array.
    """
    X = np.asarray(X)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    rng = np.random.default_rng(rng)
    idx = rng.choice(N, size=M, replace=False)
    return Points(X[idx])


def kmeans_init(X, M, rng=None):
    """k-means++ centroids of ``X`` as inducing points.

    Uses :func:`scipy.cluster.vq.kmeans2` with ``minit="++"``.

    Parameters
    ----------
    X : array-like, shape (N, D)
    M : int
        Number of clusters / inducing points.
    rng : int or numpy Generator, optional

    Returns
    -------
    Points
        Wrapping an ``(M, D)`` numpy array of centroids.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    seed = int(np.random.default_rng(rng).integers(0, 2**31 - 1))
    centroids, _ = scipy.cluster.vq.kmeans2(X, M, minit="++", seed=seed)
    return Points(centroids)


def greedy_variance_init(X, M, kernel, threshold=0.0, jitter=1e-12, rng=None, compile_kwargs=None):
    """Greedy conditional-variance (pivoted-Cholesky) selection.

    Implements the "ConditionalVariance" initialization of Burt et al. (2020),
    *Convergence of Sparse Variational Inference in GP Regression*. At each
    step, the next inducing point is the row of ``X`` with largest remaining
    conditional variance given the already-selected points — equivalent to
    running a partial pivoted Cholesky decomposition of ``K(X, X)`` with the
    standard max-diagonal pivot rule. Selected points are a **subset of X**;
    this is discrete subset selection, not continuous optimization.

    Adapted from markvdw/RobustGP (Apache-2.0). Time O(N·M^2), memory O(N·M).

    Recommended workflow
    --------------------
    Burt et al. show that with a good greedy initialization, ``Z`` typically
    does **not** need to be gradient-optimized during training — for most
    problems the frozen subset is within noise of jointly-optimized ``Z`` at a
    tiny fraction of the compute. The standard recipe:

    1. Initialize ``Z`` with ``greedy_variance_init(X, M, kernel)`` using
       initial kernel hyperparameters.
    2. Freeze ``Z``. Train the kernel/likelihood hyperparameters (and, for
       SVGP, the variational parameters).
    3. *Optional.* Re-initialize ``Z`` with the learned hyperparameters and
       retrain briefly. Usually a small improvement.

    For VFE/SGPR (Titsias collapsed bound) ``Z`` is sometimes still optimized
    because gradients are cheap; for SVGP, frozen greedy ``Z`` is the norm.

    Parameters
    ----------
    X : array-like, shape (N, D)
    M : int
        Maximum number of inducing points. Fewer may be returned if the
        approximation converges.
    kernel : Kernel
        PTGP kernel, compiled internally via ``pytensor.function``.
    threshold : float, optional
        Stop early if the trace of the residual ``K - Q`` drops below this.
        Default 0 (run the full ``M`` iterations).
    jitter : float, optional
        Small diagonal jitter for numerical stability.
    rng : int or numpy Generator, optional
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function`` when
        compiling the kernel evaluations. Use to set ``mode``
        (e.g. ``"NUMBA"``, ``"JAX"``). Same pattern as ``pm.sample``'s
        ``compile_kwargs``.

    Returns
    -------
    Points
        Wrapping an ``(M', D)`` numpy array with ``M' <= M``.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a ptgp Kernel")
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    rng = np.random.default_rng(rng)

    D = X.shape[1]
    X_sym = pt.matrix("_X", shape=(None, D), dtype="float64")
    Y_sym = pt.matrix("_Y", shape=(None, D), dtype="float64")
    ck = compile_kwargs or {}
    k_cross_fn = pytensor.function([X_sym, Y_sym], kernel(X_sym, Y_sym), **ck)
    k_diag_fn = pytensor.function([X_sym], pt.diag(kernel(X_sym)), **ck)

    perm = rng.permutation(N)
    Xp = X[perm]

    d = k_diag_fn(Xp) + jitter
    indices = np.zeros(M, dtype=int)
    indices[0] = int(np.argmax(d))

    if M == 1:
        return Points(Xp[indices])

    C = np.zeros((M - 1, N))
    final_m = M

    for m in range(M - 1):
        j = int(indices[m])
        dj = np.sqrt(d[j])
        cj = C[:m, j]

        Kj = k_cross_fn(Xp, Xp[j : j + 1]).ravel()
        Kj[j] += jitter

        e = (Kj - C[:m].T @ cj) / dj
        C[m, :] = e

        d = np.maximum(d - e**2, 0.0)

        indices[m + 1] = int(np.argmax(d))

        if d.sum() < threshold:
            final_m = m + 2
            break

    return Points(Xp[indices[:final_m]])