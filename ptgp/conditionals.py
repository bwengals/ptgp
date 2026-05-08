import pytensor.tensor as pt


# Diagonal jitter added to Kmm before Cholesky / inversion, to keep it PSD
# under floating-point noise. Matches GPflow / GPJax / PyMC defaults of 1e-6.
_DEFAULT_JITTER = 1e-6


def base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False, full_cov=False):
    """Compute the posterior conditional q(f*) = integral q(u) p(f*|u) du.

    Given:
        p(u) = N(0, Kmm)
        p(f*|u) = N(Kmn.T @ Kmm^{-1} @ u,  Knn - Kmn.T @ Kmm^{-1} @ Kmn)
        q(u) = N(f, q_sqrt @ q_sqrt.T)

    Parameters
    ----------
    Kmn : tensor, shape (M, N)
        Cross-covariance between inducing and prediction points.
    Kmm : tensor, shape (M, M)
        Inducing point covariance (annotated PSD).
    Knn : tensor
        If ``full_cov=False``, shape (N,), the diagonal of the prediction
        covariance. If ``full_cov=True``, shape (N, N), the full prediction
        covariance.
    f : tensor, shape (M,)
        Variational mean (or posterior mean for exact GP).
    q_sqrt : tensor, shape (M, M), optional
        Cholesky factor of variational covariance. If None, returns the
        prior conditional (used by exact GP).
    white : bool
        If True, f and q_sqrt are in the whitened parameterization
        (prior on v is N(0, I) instead of N(0, Kmm)).
    full_cov : bool
        If True, return the full (N, N) posterior covariance; otherwise
        return the (N,) marginal variance.

    Returns
    -------
    fmean : tensor, shape (N,)
        Posterior mean.
    fvar : tensor
        Posterior variance — shape (N,) if ``full_cov=False``, (N, N) otherwise.
    """
    # Add jitter to keep Kmm PSD under float noise — matches GPflow / PyMC default.
    # Re-annotate after the addition: PyTensor canonicalizes ``Kmm + c·I`` into a
    # ``set_subtensor`` on the diagonal, which our PSD-inference rules don't see
    # through. The mathematical identity (PSD + c·I PSD ⇒ PSD) is sound.
    Kmm = pt.assume(
        Kmm + _DEFAULT_JITTER * pt.eye(Kmm.shape[-1], dtype=Kmm.dtype),
        positive_definite=True, symmetric=True,
    )
    Kmm_inv = pt.linalg.inv(Kmm)

    # A = Kmm^{-1} @ Kmn, shape (M, N)
    A = Kmm_inv @ Kmn

    if white:
        Lmm = pt.linalg.cholesky(Kmm)
        # A_white = Lmm^{-1} @ Kmn
        A_white = pt.linalg.solve(Lmm, Kmn)
        # pt.dot(f, A_white): contracts over M (static), leaves N (dynamic) trailing.
        # Avoids DimShuffle(A_white) which makes N the leading dim and breaks JAX
        # gradient tracing (jnp.broadcast_to rejects dynamic leading shapes).
        fmean = pt.dot(f, A_white)
        if full_cov:
            fvar = Knn - pt.tensordot(A_white, A_white, [[0], [0]])
        else:
            fvar = Knn - pt.sum(A_white**2, axis=0)
    else:
        fmean = pt.dot(f, A)
        if full_cov:
            fvar = Knn - pt.tensordot(A, Kmn, [[0], [0]])
        else:
            fvar = Knn - pt.sum(A * Kmn, axis=0)

    if q_sqrt is not None:
        # B = q_sqrt.T @ A_white: shape (M, N) — M static as leading dim.
        # Previously written as A_white.T @ q_sqrt (N, M) which broke JAX gradient
        # tracing for the same reason as fmean above.
        if white:
            B = q_sqrt.T @ A_white  # (M, N)
        else:
            B = q_sqrt.T @ A  # (M, N)
        if full_cov:
            fvar = fvar + pt.tensordot(B, B, [[0], [0]])
        else:
            fvar = fvar + pt.sum(B**2, axis=0)

    return fmean, fvar
