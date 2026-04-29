import pytensor.tensor as pt

from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero


class VFE:
    """Variational Free Energy (SGPR) sparse Gaussian Process.

    Uses Titsias' collapsed bound — inducing variables are analytically
    integrated out. The observation model is Gaussian; parameterize the noise
    via ``sigma``.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: ``Zero()``).
    sigma : tensor or PyMC random variable
        Observation noise standard deviation.
    inducing_variable : InducingVariables
        Inducing point locations.
    """

    def __init__(self, kernel, mean=None, sigma=None, inducing_variable=None):
        """Store the kernel, mean, and inducing variable; build a Gaussian likelihood from sigma."""
        if not hasattr(inducing_variable, "Z"):
            raise TypeError(
                f"VFE requires inducing variables with a .Z attribute "
                f"(got {type(inducing_variable).__name__}). "
                f"Use SVGP for structured inducing variables like FourierFeatures1D."
            )
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = Gaussian(sigma)
        self.inducing_variable = inducing_variable

    def predict_marginal(self, X_new, X_train, y_train, incl_lik=False):
        """Posterior marginal mean and variance at each point in X_new.

        Returns the per-point posterior; correlations between test points
        are discarded.

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        X_train : tensor, shape (N, D)
        y_train : tensor, shape (N,)
        incl_lik : bool
            If True, include likelihood noise in the predictions.

        Returns
        -------
        mean : tensor, shape (N*,)
        var : tensor, shape (N*,)
        """
        Z = self.inducing_variable.Z
        sigma2 = self.likelihood.sigma**2

        Kuu = self.kernel(Z)  # (M, M)
        Kuf = self.kernel(Z, X_train)  # (M, N)
        Kus = self.kernel(Z, X_new)  # (M, N*)
        Kss_diag = self.kernel.diag(X_new)

        # Sigma = Kuu + Kuf @ Kuf.T / sigma^2
        Sigma = Kuu + Kuf @ Kuf.T / sigma2
        Sigma_inv = pt.linalg.inv(Sigma)

        mu_train = self.mean(X_train)
        alpha = Sigma_inv @ Kuf @ (y_train - mu_train) / sigma2

        fmean = self.mean(X_new) + Kus.T @ alpha

        # fvar = Kss - Kus.T @ (Kuu^{-1} - Sigma^{-1}) @ Kus
        Kuu_inv = pt.linalg.inv(Kuu)
        diff_inv = Kuu_inv - Sigma_inv
        fvar = Kss_diag - pt.sum(Kus * (diff_inv @ Kus), axis=0)

        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar
