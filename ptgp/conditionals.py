import pytensor.tensor as pt


def conditional_whitened(A_white, Knn, f, q_sqrt=None, full_cov=False):
    """Posterior conditional in whitened parameterisation.

    A_white : (M, N), satisfies A_white = R^{-1} @ Kmn where R @ R.T = Kmm.
    Knn     : (N,) prior diagonal if ``full_cov=False``, else (N, N) prior covariance.
    f       : (M,) variational mean (whitened: prior on v is N(0, I)).
    q_sqrt  : (M, K), optional.
    full_cov: if True, return full (N, N) covariance; else (N,) marginals.
    """
    fmean = A_white.T @ f
    if full_cov:
        fvar = Knn - A_white.T @ A_white
    else:
        fvar = Knn - pt.sum(A_white**2, axis=0)
    if q_sqrt is not None:
        B = A_white.T @ q_sqrt
        if full_cov:
            fvar = fvar + B @ B.T
        else:
            fvar = fvar + pt.sum(B**2, axis=1)
    return fmean, fvar


def conditional_unwhitened(A, Kmn, Knn, f, q_sqrt=None, full_cov=False):
    """Posterior conditional in unwhitened parameterisation.

    A    : (M, N), satisfies A = Kmm^{-1} @ Kmn.
    Kmn  : (M, N) cross-covariance.
    Knn  : (N,) prior diagonal if ``full_cov=False``, else (N, N) prior covariance.
    f    : (M,) variational mean (prior on u is N(0, Kmm)).
    full_cov: if True, return full (N, N) covariance; else (N,) marginals.
    """
    fmean = A.T @ f
    if full_cov:
        fvar = Knn - A.T @ Kmn
    else:
        fvar = Knn - pt.sum(A * Kmn, axis=0)
    if q_sqrt is not None:
        B = A.T @ q_sqrt
        if full_cov:
            fvar = fvar + B @ B.T
        else:
            fvar = fvar + pt.sum(B**2, axis=1)
    return fmean, fvar


def base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False, full_cov=False):
    """Back-compat wrapper. Materialises Kmm; use the helpers above directly
    when you have a structured Kuu_solve / Kuu_sqrt_solve."""
    if white:
        L = pt.linalg.cholesky(Kmm)
        A_white = pt.linalg.solve(L, Kmn)
        return conditional_whitened(A_white, Knn, f, q_sqrt, full_cov=full_cov)
    A = pt.linalg.inv(Kmm) @ Kmn
    return conditional_unwhitened(A, Kmn, Knn, f, q_sqrt, full_cov=full_cov)
