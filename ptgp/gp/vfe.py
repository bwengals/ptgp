import pytensor.tensor as pt

from ptgp.mean import Zero


class VFE:
    """Variational Free Energy (SGPR) sparse Gaussian Process.

    Uses Titsias' collapsed bound — inducing variables are analytically
    integrated out.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: Zero()).
    likelihood : Gaussian
        Gaussian likelihood (VFE requires Gaussian likelihood).
    inducing_variable : InducingVariables
        Inducing point locations.
    """

    def __init__(self, kernel, mean=None, likelihood=None, inducing_variable=None):
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable

    def predict(self, X_new, X_train=None, y_train=None, incl_lik=False):
        """Posterior predictive mean and variance.

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        X_train : tensor, shape (N, D), optional
        y_train : tensor, shape (N,), optional
        incl_lik : bool
            If True, include likelihood noise in the predictions.

        Returns
        -------
        mean : tensor, shape (N*,)
        var : tensor, shape (N*,)
        """
        if X_train is None:
            X_train = self._X_train
            y_train = self._y_train

        Z = self.inducing_variable.Z
        sigma2 = self.likelihood.sigma**2

        Kuu = self.kernel(Z)  # (M, M)
        Kuf = self.kernel(Z, X_train)  # (M, N)
        Kus = self.kernel(Z, X_new)  # (M, N*)
        Kss_diag = self.kernel_diag(X_new)

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

    def kernel_diag(self, X):
        """Diagonal of K(X, X)."""
        return pt.diag(self.kernel(X))
