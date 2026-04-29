import pytensor.tensor as pt

from ptgp.conditionals import conditional_unwhitened, conditional_whitened
from ptgp.kl import gauss_kl, gauss_kl_structured
from ptgp.mean import Zero


class SVGP:
    """Stochastic Variational Gaussian Process.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: Zero()).
    likelihood : Likelihood
        Observation likelihood.
    inducing_variable : InducingVariables
        Inducing point locations (or structured inducing variables).
    whiten : bool
        If True, use whitened variational parameterization (default True).
    q_mu : tensor, shape (M,), optional
        Initial variational mean. Defaults to zeros.
    q_sqrt : tensor, shape (M, M), optional
        Initial variational Cholesky factor. Defaults to identity.
    """

    def __init__(
        self,
        kernel,
        mean=None,
        likelihood=None,
        inducing_variable=None,
        whiten=True,
        q_mu=None,
        q_sqrt=None,
    ):
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.whiten = whiten

        M = inducing_variable.num_inducing
        self.q_mu = q_mu if q_mu is not None else pt.zeros(M)
        if q_sqrt is None:
            self.q_sqrt = pt.specify_assumptions(pt.eye(M), diagonal=True)
        else:
            self.q_sqrt = pt.specify_assumptions(q_sqrt, lower_triangular=True)

    def predict_marginal(self, X, incl_lik=False):
        """Posterior marginal mean and variance at each point in X.

        Returns the per-point posterior — correlations between test
        points are discarded. Use ``predict_joint`` for the full (N, N)
        covariance or ``predict_f_samples`` to draw smooth function samples.

        Parameters
        ----------
        X : tensor, shape (N, D)
        incl_lik : bool
            If True, push through the likelihood's predictive mean/var
            (observation-space uncertainty).

        Returns
        -------
        mean : tensor, shape (N,)
        var : tensor, shape (N,)
        """
        ind, kernel = self.inducing_variable, self.kernel
        Kmn = ind.K_uf(kernel, X)
        Knn_diag = kernel.diag(X)
        if self.whiten:
            A_w = ind.Kuu_sqrt_solve(kernel, Kmn)
            fmean, fvar = conditional_whitened(A_w, Knn_diag, self.q_mu, self.q_sqrt)
        else:
            A = ind.Kuu_solve(kernel, Kmn)
            fmean, fvar = conditional_unwhitened(A, Kmn, Knn_diag, self.q_mu, self.q_sqrt)
        fmean = fmean + self.mean(X)
        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar

    def predict_joint(self, X):
        """Posterior joint mean and full covariance of the latent f at X.

        The diagonal of the returned covariance equals the variance from
        ``predict_marginal``; the off-diagonals capture correlations
        between test points, which are needed to draw smooth samples.

        Parameters
        ----------
        X : tensor, shape (N, D)

        Returns
        -------
        mean : tensor, shape (N,)
        cov : tensor, shape (N, N)
        """
        ind, kernel = self.inducing_variable, self.kernel
        Kmn = ind.K_uf(kernel, X)
        Knn = kernel(X)
        if self.whiten:
            A_w = ind.Kuu_sqrt_solve(kernel, Kmn)
            fmean, fcov = conditional_whitened(A_w, Knn, self.q_mu, self.q_sqrt, full_cov=True)
        else:
            A = ind.Kuu_solve(kernel, Kmn)
            fmean, fcov = conditional_unwhitened(A, Kmn, Knn, self.q_mu, self.q_sqrt, full_cov=True)
        fmean = fmean + self.mean(X)
        return fmean, fcov

    def predict_f_samples(self, X, epsilon, jitter=1e-6):
        """Draw samples of the latent f at X from the joint posterior.

        Samples are produced via a Cholesky transform of caller-supplied
        iid-standard-normal noise. The caller owns the RNG — the function
        itself is deterministic.

        Parameters
        ----------
        X : tensor, shape (N, D)
        epsilon : tensor, shape (S, N)
            iid N(0, 1) draws. ``S`` is the number of samples.
        jitter : float
            Added to the diagonal of the posterior covariance before
            Cholesky, for numerical stability (default 1e-6).

        Returns
        -------
        samples : tensor, shape (S, N)
        """
        fmean, fcov = self.predict_joint(X)
        N = X.shape[0]
        L = pt.linalg.cholesky(fcov + jitter * pt.eye(N))
        return fmean[None, :] + epsilon @ L.T

    def prior_kl(self):
        """KL divergence KL[q(u) || p(u)]."""
        if self.whiten:
            return gauss_kl(self.q_mu, self.q_sqrt, K=None)
        ind, kernel = self.inducing_variable, self.kernel
        return gauss_kl_structured(
            self.q_mu,
            self.q_sqrt,
            K_solve=lambda rhs: ind.Kuu_solve(kernel, rhs),
            K_logdet=ind.Kuu_logdet(kernel),
        )
