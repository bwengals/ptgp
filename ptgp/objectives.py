import pytensor.tensor as pt

from ptgp.kl import gauss_kl


def marginal_log_likelihood(gp, X, y):
    """Exact GP log marginal likelihood.

    log p(y|X, theta) = log N(y; m(X), K(X,X) + sigma^2 I)

    Parameters
    ----------
    gp : GP
        Exact GP model with kernel, mean function, and likelihood.
    X : tensor, shape (N, D)
    y : tensor, shape (N,)

    Returns
    -------
    scalar
        Log marginal likelihood.
    """
    mu = gp.mean(X)
    K = gp.kernel(X) + gp.likelihood.sigma**2 * pt.eye(X.shape[0])

    diff = y - mu
    sign, logdet = pt.linalg.slogdet(K)
    K_inv = pt.linalg.inv(K)
    N = X.shape[0]

    return -0.5 * (diff @ K_inv @ diff + logdet + N * pt.log(2.0 * pt.pi))


def elbo(svgp, X, y, n_data=None):
    """SVGP evidence lower bound.

    ELBO = E_{q(f)}[log p(y|f)] - KL[q(u) || p(u)]

    Scaled by n_data / batch_size for minibatch training.

    Parameters
    ----------
    svgp : SVGP
        Stochastic variational GP model.
    X : tensor, shape (batch_size, D)
    y : tensor, shape (batch_size,)
    n_data : int, optional
        Total number of data points. If None, no scaling is applied.

    Returns
    -------
    scalar
        ELBO value.
    """
    fmean, fvar = svgp.predict_f(X)

    var_exp = svgp.likelihood.variational_expectation(y, fmean, fvar)
    var_exp_sum = pt.sum(var_exp)

    if n_data is not None:
        batch_size = X.shape[0]
        scale = n_data / batch_size
    else:
        scale = 1.0

    kl = svgp.prior_kl()

    return scale * var_exp_sum - kl


def collapsed_elbo(vfe, X, y):
    """VFE/SGPR collapsed ELBO (Titsias' bound).

    The inducing variables are analytically integrated out, giving a
    bound on the exact marginal likelihood.

    ELBO = log N(y; m, Q + sigma^2 I) - 1/(2*sigma^2) * tr(K - Q)

    where Q = Kuf.T @ Kuu^{-1} @ Kuf is the Nystrom approximation.

    Parameters
    ----------
    vfe : VFE
        VFE sparse GP model.
    X : tensor, shape (N, D)
    y : tensor, shape (N,)

    Returns
    -------
    scalar
        Collapsed ELBO value.
    """
    sigma2 = vfe.likelihood.sigma**2
    N = X.shape[0]
    Z = vfe.inducing_variable.Z

    mu = vfe.mean(X)
    Kff_diag = vfe.kernel_diag(X)
    Kuf = vfe.kernel(Z, X)  # (M, N)
    Kuu = vfe.kernel(Z)     # (M, M)

    Kuu_inv = pt.linalg.inv(Kuu)

    # Nystrom approximation Q = Kuf.T @ Kuu^{-1} @ Kuf
    # Q_diag = diag(Kuf.T @ Kuu^{-1} @ Kuf)
    A = Kuu_inv @ Kuf  # (M, N)
    Q_diag = pt.sum(A * Kuf, axis=0)  # (N,)

    # Effective covariance: Q + sigma^2 I
    # For the log-likelihood term, we need log N(y; mu, Q + sigma^2 I)
    Q_mat = Kuf.T @ A  # (N, N)
    cov = Q_mat + sigma2 * pt.eye(N)

    diff = y - mu
    sign, logdet = pt.linalg.slogdet(cov)
    cov_inv = pt.linalg.inv(cov)

    # log N(y; mu, Q + sigma^2 I)
    fit = -0.5 * (diff @ cov_inv @ diff + logdet + N * pt.log(2.0 * pt.pi))

    # Trace penalty: -1/(2*sigma^2) * tr(Kff - Q)
    trace_penalty = -0.5 / sigma2 * pt.sum(Kff_diag - Q_diag)

    return fit + trace_penalty
