import pytensor.tensor as pt

from ptgp.kernels.base import Kernel


class RandomWalk(Kernel):
    """Random walk (Brownian motion / Wiener process) kernel.

    k(x, y) = min(x, y)

    Defined for positive inputs. For 1D input only.
    Scale with multiplication: eta**2 * RandomWalk()
    """

    input_dim = 1

    def __call__(self, X, Y=None):
        symmetric = Y is None
        if symmetric:
            Y = X
        K = pt.minimum(X, Y.T)
        if symmetric:
            K = pt.specify_assumptions(K, symmetric=True, positive_definite=True)
        return K
