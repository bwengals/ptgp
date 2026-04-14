import pytensor.tensor as pt


class Zero:
    """Zero mean function."""

    def __call__(self, X):
        return pt.zeros(X.shape[0])


class Constant:
    """Constant mean function m(x) = c.

    Parameters
    ----------
    c : tensor or PyMC random variable
    """

    def __init__(self, c):
        self.c = c

    def __call__(self, X):
        return pt.ones(X.shape[0]) * self.c


class Linear:
    """Linear mean function m(x) = x @ coeffs + intercept.

    Parameters
    ----------
    coeffs : tensor or PyMC random variable, shape (d,)
    intercept : tensor or PyMC random variable, optional
    """

    def __init__(self, coeffs, intercept=0.0):
        self.coeffs = coeffs
        self.intercept = intercept

    def __call__(self, X):
        return X @ self.coeffs + self.intercept
