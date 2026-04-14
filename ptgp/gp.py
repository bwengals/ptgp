import pytensor.tensor as pt

from ptgp.conditionals import base_conditional
from ptgp.mean import Zero


class GP:
    """Exact (unapproximated) Gaussian process.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: Zero()).
    likelihood : Gaussian
        Gaussian likelihood (exact GP requires Gaussian likelihood).
    """

    def __init__(self, kernel, mean=None, likelihood=None):
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = likelihood
        self._X_train = None
        self._y_train = None

    def predict_f(self, X_new, X_train=None, y_train=None):
        """Posterior predictive mean and variance of the latent function.

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        X_train : tensor, shape (N, D), optional
            Uses stored training data if not provided.
        y_train : tensor, shape (N,), optional

        Returns
        -------
        fmean : tensor, shape (N*,)
        fvar : tensor, shape (N*,)
        """
        if X_train is None:
            X_train = self._X_train
            y_train = self._y_train

        Knn = self.kernel(X_train)
        Knn_noisy = Knn + self.likelihood.sigma**2 * pt.eye(X_train.shape[0])
        Kns = self.kernel(X_train, X_new)  # (N, N*)
        Kss_diag = self.kernel_diag(X_new)

        mu_train = self.mean(X_train)
        alpha = pt.linalg.solve(Knn_noisy, y_train - mu_train)

        fmean = self.mean(X_new) + Kns.T @ alpha
        v = pt.linalg.solve(Knn_noisy, Kns)
        fvar = Kss_diag - pt.sum(Kns * v, axis=0)

        return fmean, fvar

    def kernel_diag(self, X):
        """Diagonal of K(X, X)."""
        return pt.diag(self.kernel(X))
