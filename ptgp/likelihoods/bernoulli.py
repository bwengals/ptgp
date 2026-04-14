import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood


def inv_probit(x):
    """Probit link: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), clamped to (jitter, 1-jitter)."""
    jitter = 1e-3
    return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0))) * (1.0 - 2.0 * jitter) + jitter


class Bernoulli(Likelihood):
    """Bernoulli likelihood: p(y=1|f) = invlink(f).

    Default link is probit. Variational expectation via Gauss-Hermite quadrature.

    Parameters
    ----------
    invlink : callable, optional
        Inverse link function (default: probit). Use ``pt.sigmoid`` for logit link.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    def __init__(self, invlink=None, n_points=20):
        self.invlink = invlink or inv_probit
        self.n_points = n_points

    def _log_prob(self, f, y):
        p = self.invlink(f)
        return y * pt.log(p) + (1.0 - y) * pt.log(1.0 - p)

    def _conditional_mean(self, f):
        return self.invlink(f)

    def _conditional_variance(self, f):
        p = self.invlink(f)
        return p * (1.0 - p)

    def predict_mean_and_var(self, mu, var):
        """Closed-form for probit link: p = Phi(mu / sqrt(1 + var)).

        Falls back to quadrature for other link functions.
        """
        if self.invlink is inv_probit:
            p = inv_probit(mu / pt.sqrt(1.0 + var))
            return p, p - p**2
        return super().predict_mean_and_var(mu, var)
