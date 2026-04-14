import pytensor.tensor as pt


def gauss_kl(q_mu, q_sqrt, K=None):
    """KL divergence KL[q || p] between multivariate Gaussians.

    q(x) = N(q_mu, q_sqrt @ q_sqrt.T)
    p(x) = N(0, K)   if K is provided (unwhitened)
    p(x) = N(0, I)    if K is None (whitened)

    Parameters
    ----------
    q_mu : tensor, shape (M,)
        Variational mean.
    q_sqrt : tensor, shape (M, M)
        Lower-triangular Cholesky factor of variational covariance.
    K : tensor, shape (M, M), optional
        Prior covariance. If None, prior is N(0, I).

    Returns
    -------
    scalar
        KL divergence.
    """
    M = q_mu.shape[0]
    q_cov = q_sqrt @ q_sqrt.T

    if K is None:
        # Whitened: KL[N(q_mu, q_cov) || N(0, I)]
        # = 0.5 * (tr(q_cov) + q_mu.T @ q_mu - M - log|q_cov|)
        trace = pt.trace(q_cov)
        mahal = q_mu @ q_mu
        sign, logdet = pt.linalg.slogdet(q_cov)
        return 0.5 * (trace + mahal - M - logdet)
    else:
        # Unwhitened: KL[N(q_mu, q_cov) || N(0, K)]
        # = 0.5 * (tr(K^{-1} q_cov) + q_mu.T @ K^{-1} @ q_mu - M - log|q_cov| + log|K|)
        K_inv = pt.linalg.inv(K)
        trace = pt.trace(K_inv @ q_cov)
        mahal = q_mu @ K_inv @ q_mu
        sign_q, logdet_q = pt.linalg.slogdet(q_cov)
        sign_K, logdet_K = pt.linalg.slogdet(K)
        return 0.5 * (trace + mahal - M - logdet_q + logdet_K)
