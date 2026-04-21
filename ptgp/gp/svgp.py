import pytensor.tensor as pt

from ptgp.conditionals import base_conditional
from ptgp.kl import gauss_kl
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
        Inducing point locations.
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
        self.q_sqrt = q_sqrt if q_sqrt is not None else pt.eye(M)

    def predict(self, X, incl_lik=False):
        """Posterior predictive mean and variance.

        Parameters
        ----------
        X : tensor, shape (N, D)
        incl_lik : bool
            If True, include likelihood noise in the predictions.

        Returns
        -------
        mean : tensor, shape (N,)
        var : tensor, shape (N,)
        """
        Z = self.inducing_variable.Z
        Kmm = self.kernel(Z)
        Kmn = self.kernel(Z, X)
        Knn_diag = self.kernel_diag(X)

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn_diag, self.q_mu, self.q_sqrt, white=self.whiten
        )
        fmean = fmean + self.mean(X)
        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar

    def prior_kl(self):
        """KL divergence KL[q(u) || p(u)]."""
        if self.whiten:
            return gauss_kl(self.q_mu, self.q_sqrt, K=None)
        else:
            Kuu = self.kernel(self.inducing_variable.Z)
            return gauss_kl(self.q_mu, self.q_sqrt, K=Kuu)

    def kernel_diag(self, X):
        """Diagonal of K(X, X)."""
        return pt.diag(self.kernel(X))
