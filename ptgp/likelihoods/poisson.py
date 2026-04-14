import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood


class Poisson(Likelihood):
    """Poisson likelihood: p(y|f) = Poisson(y; invlink(f)).

    Default link is log (invlink=exp). Has a closed-form variational
    expectation with the log link; falls back to quadrature for other links.

    Parameters
    ----------
    invlink : callable, optional
        Inverse link function (default: exp).
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    def __init__(self, invlink=None, n_points=20):
        self.invlink = invlink or pt.exp
        self.n_points = n_points

    def _log_prob(self, f, y):
        lam = self.invlink(f)
        return y * pt.log(lam) - lam - pt.gammaln(y + 1.0)

    def _conditional_mean(self, f):
        return self.invlink(f)

    def _conditional_variance(self, f):
        return self.invlink(f)

    def variational_expectation(self, y, mu, var):
        """Closed-form for log link: E_q[y*f - exp(f) - log(y!)].

        Falls back to quadrature for other link functions.
        """
        if self.invlink is pt.exp:
            return y * mu - pt.exp(mu + var / 2.0) - pt.gammaln(y + 1.0)
        return super().variational_expectation(y, mu, var)
