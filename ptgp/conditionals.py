import pytensor.tensor as pt


def base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False):
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
    Knn : tensor, shape (N,)
        Diagonal of prediction point covariance.
    f : tensor, shape (M,)
        Variational mean (or posterior mean for exact GP).
    q_sqrt : tensor, shape (M, M), optional
        Cholesky factor of variational covariance. If None, returns the
        prior conditional (used by exact GP).
    white : bool
        If True, f and q_sqrt are in the whitened parameterization
        (prior on v is N(0, I) instead of N(0, Kmm)).

    Returns
    -------
    fmean : tensor, shape (N,)
        Posterior mean.
    fvar : tensor, shape (N,)
        Posterior variance (diagonal).
    """
    Kmm_inv = pt.linalg.inv(Kmm)

    # A = Kmm^{-1} @ Kmn, shape (M, N)
    A = Kmm_inv @ Kmn

    # Prior conditional mean and variance
    # fmean = Kmn.T @ Kmm^{-1} @ f = A.T @ f
    # fvar = Knn - diag(Kmn.T @ Kmm^{-1} @ Kmn) = Knn - diag(A.T @ Kmn)
    if white:
        # In whitened parameterization, u = Lmm @ v where Lmm = chol(Kmm).
        # p(f*|v) = N(Kmn.T @ Kmm^{-1} @ Lmm @ v, ...)
        # Since Kmm = Lmm @ Lmm.T, Kmm^{-1} @ Lmm = Lmm^{-T}
        # So the mean becomes Kmn.T @ Lmm^{-T} @ v
        # We need: fmean = Kmn.T @ Lmm^{-T} @ f
        # But we can write Lmm^{-T} = Kmm^{-1} @ Lmm
        # Equivalently: fmean = (Lmm^{-1} @ Kmn).T @ f
        # Using naive linalg: Lmm = chol(Kmm), but we want to stay naive.
        # Actually in whitened coords: the mapping is through Lmm.
        # Let's compute Lmm and use it.
        Lmm = pt.linalg.cholesky(Kmm)
        # A_white = Lmm^{-1} @ Kmn
        A_white = pt.linalg.solve(Lmm, Kmn)
        fmean = A_white.T @ f
        fvar = Knn - pt.sum(A_white**2, axis=0)
    else:
        fmean = A.T @ f
        fvar = Knn - pt.sum(A * Kmn, axis=0)

    if q_sqrt is not None:
        # Add variational uncertainty
        # Var contribution = diag(A.T @ q_cov @ A) where q_cov = q_sqrt @ q_sqrt.T
        if white:
            B = A_white.T @ q_sqrt  # (N, M)
        else:
            B = A.T @ q_sqrt  # (N, M)
        fvar = fvar + pt.sum(B**2, axis=1)

    return fmean, fvar
