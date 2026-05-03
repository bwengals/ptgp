"""Inducing variables and initialization strategies.

The `*_init` functions return an :class:`Points` wrapping a plain
numpy array, so ``ip.Z`` is directly usable for plotting.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.cluster.vq
from dataclasses import dataclass

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


def kmeans_init(X, M, rng=None, tol=1e-6):
    """k-means++ centroids of ``X`` as inducing points.

    Uses :func:`scipy.cluster.vq.kmeans2` with ``minit="++"``.  After
    clustering, any centroids whose pairwise Euclidean distance is below
    ``tol`` are deduplicated (greedy: the first of each near-duplicate group
    is kept).  A summary is printed if any are removed.

    Parameters
    ----------
    X : array-like, shape (N, D)
    M : int
        Number of clusters / inducing points requested.
    rng : int or numpy Generator, optional
    tol : float, optional
        Euclidean-distance threshold below which two centroids are considered
        duplicates.  Default ``1e-6``.

    Returns
    -------
    Points
        Wrapping an ``(M', D)`` numpy array of centroids, with ``M' <= M``.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    seed = int(np.random.default_rng(rng).integers(0, 2**31 - 1))
    centroids, _ = scipy.cluster.vq.kmeans2(X, M, minit="++", seed=seed)

    # Greedy deduplication: keep centroid i if it is > tol from all earlier
    # kept centroids.  O(M^2) but M is typically small (100-500).
    dists = np.sqrt(np.sum((centroids[:, None] - centroids[None, :]) ** 2, axis=-1))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        if not keep[i]:
            continue
        near = dists[i, i + 1 :] < tol
        keep[i + 1 :][near] = False

    n_removed = int((~keep).sum())
    if n_removed > 0:
        print(
            f"kmeans_init: removed {n_removed} near-duplicate centroid(s) "
            f"(tol={tol:.0e}); returning {keep.sum()} of {M} requested."
        )

    return Points(centroids[keep])


@dataclass
class GreedyVarianceDiagnostics:
    """Diagnostics returned by :func:`greedy_variance_init`.

    Attributes
    ----------
    trace_curve : ndarray, shape (M,)
        Remaining unexplained variance after conditioning on each successive
        inducing point. ``trace_curve[0]`` is the total kernel diagonal sum
        before any conditioning; ``trace_curve[m]`` is the residual after
        conditioning on ``m`` points. Divide by ``total_variance`` to get the
        fraction of variance still unexplained.
    d_final : ndarray, shape (N,)
        Per-data-point residual conditional variance after all M points are
        selected. Large values identify data points poorly covered by the
        current inducing set.
    total_variance : float
        Total kernel diagonal sum before conditioning (``trace_curve[0]``).
    kuu_min_eigenvalue : float
        Smallest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_max_eigenvalue : float
        Largest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_condition_number : float
        ``max_eig / min_eig``. Values below ~1e5 are numerically healthy.
        Large values indicate near-duplicate inducing points or a kernel
        lengthscale too short relative to the inducing-point spacing.
    kuu_n_small_eigenvalues : int
        Number of eigenvalues below ``kuu_eig_threshold``. Non-zero values
        mean the inducing-point covariance is near-singular.
    kuu_eig_threshold : float
        Threshold used to count small eigenvalues (default 1e-4).
    """
    trace_curve: np.ndarray
    d_final: np.ndarray
    total_variance: float
    kuu_min_eigenvalue: float
    kuu_max_eigenvalue: float
    kuu_condition_number: float
    kuu_n_small_eigenvalues: int
    kuu_eig_threshold: float

    def __repr__(self):
        M = len(self.trace_curve)
        pct = 100.0 * (1.0 - self.trace_curve[-1] / self.total_variance)
        lines = [
            f"M                 : {M}",
            f"variance explained: {pct:.1f}%",
            f"min eigenvalue    : {self.kuu_min_eigenvalue:.3g}",
            f"max eigenvalue    : {self.kuu_max_eigenvalue:.3g}",
            f"condition number  : {self.kuu_condition_number:.3g}",
            f"eigs < {self.kuu_eig_threshold:.0e}     : {self.kuu_n_small_eigenvalues}",
        ]
        return "\n".join(lines)


def greedy_variance_init(X, M, kernel, threshold=0.0, jitter=1e-12, rng=None,
                         eig_threshold=1e-4, compile_kwargs=None):
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
    eig_threshold : float, optional
        Eigenvalues of ``K(Z, Z) + jitter * I`` below this value are counted
        in ``kuu_n_small_eigenvalues``. Default ``1e-4``.
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function`` when
        compiling the kernel evaluations. Use to set ``mode``
        (e.g. ``"NUMBA"``, ``"JAX"``). Same pattern as ``pm.sample``'s
        ``compile_kwargs``.

    Returns
    -------
    points : Points
        Wrapping an ``(M', D)`` numpy array with ``M' <= M``.
    diagnostics : GreedyVarianceDiagnostics
        Dataclass with fields ``trace_curve``, ``d_final``, ``total_variance``,
        ``kuu_min_eigenvalue``, ``kuu_max_eigenvalue``, ``kuu_condition_number``,
        ``kuu_n_small_eigenvalues``, and ``kuu_eig_threshold``.
        ``repr(diagnostics)`` prints a one-screen summary.
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
    total_variance = float(d.sum())
    indices = np.zeros(M, dtype=int)
    indices[0] = int(np.argmax(d))

    if M == 1:
        Z1 = Xp[indices]
        Kuu1 = k_cross_fn(Z1, Z1) + jitter * np.eye(1)
        eig1 = float(Kuu1[0, 0])
        diag = GreedyVarianceDiagnostics(
            trace_curve=np.array([total_variance]),
            d_final=d.copy(),
            total_variance=total_variance,
            kuu_min_eigenvalue=eig1,
            kuu_max_eigenvalue=eig1,
            kuu_condition_number=1.0,
            kuu_n_small_eigenvalues=int(eig1 < eig_threshold),
            kuu_eig_threshold=eig_threshold,
        )
        return Points(Z1), diag

    C = np.zeros((M - 1, N))
    final_m = M
    trace_curve = np.empty(M)
    trace_curve[0] = total_variance

    for m in range(M - 1):
        j = int(indices[m])
        dj = np.sqrt(d[j])
        cj = C[:m, j]

        Kj = k_cross_fn(Xp, Xp[j : j + 1]).ravel()
        Kj[j] += jitter

        e = (Kj - C[:m].T @ cj) / dj
        C[m, :] = e

        d = np.maximum(d - e**2, 0.0)
        trace_curve[m + 1] = float(d.sum())

        indices[m + 1] = int(np.argmax(d))

        if d.sum() < threshold:
            final_m = m + 2
            break

    Z_selected = Xp[indices[:final_m]]
    Kuu = k_cross_fn(Z_selected, Z_selected) + jitter * np.eye(final_m)
    eigs = np.linalg.eigvalsh(Kuu)  # sorted ascending

    diag = GreedyVarianceDiagnostics(
        trace_curve=trace_curve[:final_m].copy(),
        d_final=d.copy(),
        total_variance=total_variance,
        kuu_min_eigenvalue=float(eigs[0]),
        kuu_max_eigenvalue=float(eigs[-1]),
        kuu_condition_number=float(eigs[-1] / eigs[0]),
        kuu_n_small_eigenvalues=int(np.sum(eigs < eig_threshold)),
        kuu_eig_threshold=eig_threshold,
    )
    return Points(Z_selected), diag