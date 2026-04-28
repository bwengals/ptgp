import pytensor.tensor as pt


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
    fmean, fvar = svgp.predict_marginal(X)

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
    """VFE/SGPR collapsed ELBO (Titsias' bound), Woodbury form.

    Same loss as the standard formulation
        ELBO = log N(y; m, Q + σ²I) - 1/(2σ²) * tr(Kff - Q),
    where ``Q = Kuf.T @ inv(Kuu) @ Kuf`` is the Nystrom approximation,
    but the N×N inverse and log-det of ``cov = σ²I + Q`` are rewritten
    via the Woodbury identity into operations on the M×M matrix
    ``D = σ²·Kuu + Kuf @ Kuf.T``. This is mathematically equivalent and
    much better conditioned when N >> M (the regime sparse GPs are made
    for) — the original N×N covariance has N - M eigenvalues clamped at
    exactly σ², which causes catastrophic cancellation in the gradient.

    Identities used (B = chol(Kuu)^{-1} @ Kuf, but never materialised):
        inv(σ²I + Q) = (I - Kuf.T @ inv(D) @ Kuf) / σ²
        log|σ²I + Q| = (N - M) · log σ² + log|D| - log|Kuu|

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
    M = Z.shape[0]

    mu = vfe.mean(X)
    Kff_diag = vfe.kernel.diag(X)
    Kuf = vfe.kernel(Z, X)            # M × N
    Kuu = vfe.kernel(Z)               # M × M

    # Q_diag = diag(Kuf.T @ inv(Kuu) @ Kuf) for the trace penalty.
    Kuu_inv_Kuf = pt.linalg.inv(Kuu) @ Kuf
    Q_diag = pt.sum(Kuf * Kuu_inv_Kuf, axis=0)

    # Woodbury: invert only the M × M matrix D, never the N × N cov.
    D = sigma2 * Kuu + Kuf @ Kuf.T    # PSD by Gram construction
    D = pt.specify_assumptions(D, positive_definite=True, symmetric=True)

    diff = y - mu
    Kuf_diff = Kuf @ diff
    quad = (diff @ diff - Kuf_diff @ pt.linalg.inv(D) @ Kuf_diff) / sigma2

    _, logdet_D = pt.linalg.slogdet(D)
    _, logdet_Kuu = pt.linalg.slogdet(Kuu)
    logdet_cov = (N - M) * pt.log(sigma2) + logdet_D - logdet_Kuu

    fit = -0.5 * (quad + logdet_cov + N * pt.log(2.0 * pt.pi))
    trace_penalty = -0.5 / sigma2 * pt.sum(Kff_diag - Q_diag)
    return fit + trace_penalty
