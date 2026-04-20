import pytensor.tensor as pt

from ptgp.kernels.base import Kernel


def _check_input_dim(k1, k2):
    """Ensure two kernels in a combination share the same input_dim."""
    if isinstance(k1, Kernel) and isinstance(k2, Kernel):
        if k1.input_dim != k2.input_dim:
            raise ValueError(
                f"Kernels in a combination must share input_dim "
                f"(got {k1.input_dim} and {k2.input_dim})"
            )


class SumKernel(Kernel):
    """Sum of two kernels: k(x, y) = k1(x, y) + k2(x, y)."""

    def __init__(self, k1, k2):
        _check_input_dim(k1, k2)
        self.k1 = k1
        self.k2 = k2
        if isinstance(k1, Kernel):
            self.input_dim = k1.input_dim
        elif isinstance(k2, Kernel):
            self.input_dim = k2.input_dim

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
        _check_input_dim(k1, k2)
        self.k1 = k1
        self.k2 = k2
        if isinstance(k1, Kernel):
            self.input_dim = k1.input_dim
        elif isinstance(k2, Kernel):
            self.input_dim = k2.input_dim

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