import pytensor.tensor as pt

from ptgp.kernels.base import Kernel


class SumKernel(Kernel):
    """Sum of two kernels: k(x, y) = k1(x, y) + k2(x, y)."""

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        symmetric = Y is None
        K = self.k1(X, Y) + self.k2(X, Y)
        if symmetric:
            K = pt.specify_assumptions(K, symmetric=True, positive_definite=True)
        return K


class ProductKernel(Kernel):
    """Product of two kernels: k(x, y) = k1(x, y) * k2(x, y).

    Also handles scalar * kernel via __rmul__ on the base class.
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        symmetric = Y is None
        k1_is_kernel = isinstance(self.k1, Kernel)
        k2_is_kernel = isinstance(self.k2, Kernel)
        if k1_is_kernel and k2_is_kernel:
            K = self.k1(X, Y) * self.k2(X, Y)
        elif k1_is_kernel:
            K = self.k1(X, Y) * self.k2
        else:
            K = self.k1 * self.k2(X, Y)
        if symmetric:
            K = pt.specify_assumptions(K, symmetric=True, positive_definite=True)
        return K
