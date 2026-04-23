import pytensor.tensor as pt

from ptgp.kernels.base import Kernel
from ptgp.kernels.stationary import _squared_distance


class RandomWalk(Kernel):
    """Random walk (Brownian motion / Wiener process) kernel.

    k(x, y) = min(x, y)

    Defined for positive inputs. 1D only.
    Scale with multiplication: eta**2 * RandomWalk()

    Parameters
    ----------
    active_dims : sequence of int, optional
        Column of ``X`` to operate on. Must have length 1.
    """

    def __init__(self, active_dims=None):
        """Fix ``input_dim=1`` and require ``active_dims`` of length 1."""
        super().__init__(input_dim=1, active_dims=active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("RandomWalk kernel requires active_dims of length 1")

    def _eval(self, X, Y):
        """Pairwise min over the active dim."""
        Xa = X[:, self.active_dims]
        Ya = Y[:, self.active_dims]
        return pt.minimum(Xa, Ya.T)

    def diag(self, X):
        """Diagonal of K(X, X). min(x, x) = x."""
        return X[:, self.active_dims[0]]


class Gibbs(Kernel):
    """Gibbs kernel with a location-dependent lengthscale function.

    k(x, y) = sqrt(2 * l(x) * l(y) / (l(x)^2 + l(y)^2))
             * exp(-(x - y)^2 / (l(x)^2 + l(y)^2))

    1D only.

    Parameters
    ----------
    lengthscale_func : callable
        ``lengthscale_func(X) -> tensor`` returning per-point lengthscales of
        shape ``(N,)`` for input ``X`` of shape ``(N, 1)``.
    active_dims : sequence of int, optional
        Column of ``X`` to operate on. Must have length 1.
    """

    def __init__(self, lengthscale_func, active_dims=None):
        """Fix ``input_dim=1`` and require ``active_dims`` of length 1."""
        super().__init__(input_dim=1, active_dims=active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("Gibbs kernel requires active_dims of length 1")
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")
        self.lengthscale_func = lengthscale_func

    def _eval(self, X, Y):
        """Evaluate the Gibbs kernel with per-point lengthscales."""
        Xa = X[:, self.active_dims]
        Ya = Y[:, self.active_dims]
        rx = self.lengthscale_func(Xa)
        ry = self.lengthscale_func(Ya)
        r2 = _squared_distance(Xa, Ya)
        rx2 = pt.reshape(pt.square(rx), (-1, 1))
        ry2 = pt.reshape(pt.square(ry), (1, -1))
        denom = rx2 + ry2
        return pt.sqrt(2.0 * pt.outer(rx, ry) / denom) * pt.exp(-r2 / denom)

    def diag(self, X):
        """Diagonal of K(X, X). k(x, x) = 1 for any lengthscale function."""
        return pt.ones(X.shape[0])


class WarpedInput(Kernel):
    """Warp the inputs of another kernel with an arbitrary function.

    k(x, y) = k_inner(w(x), w(y))

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` this kernel expects. May differ from
        ``kernel_func.input_dim`` — the warp may change dimensionality.
    kernel_func : Kernel
        Inner kernel applied to the warped inputs.
    warp_func : callable
        ``warp_func(X) -> X_warped``. PyTensor function.
    active_dims : sequence of int, optional
        Columns of ``X`` to warp. Defaults to all.
    """

    def __init__(self, input_dim, kernel_func, warp_func, active_dims=None):
        """Validate dimensions via the base class and store the inner kernel and warp."""
        super().__init__(input_dim, active_dims)
        if not isinstance(kernel_func, Kernel):
            raise TypeError("kernel_func must be a Kernel")
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")
        self.kernel_func = kernel_func
        self.warp_func = warp_func

    def _eval(self, X, Y):
        """Warp both inputs, then delegate to the inner kernel."""
        Xw = self.warp_func(X[:, self.active_dims])
        Yw = self.warp_func(Y[:, self.active_dims])
        return self.kernel_func(Xw, Yw)

    def diag(self, X):
        """Diagonal of K(X, X): warp the inputs, then delegate to inner kernel."""
        return self.kernel_func.diag(self.warp_func(X[:, self.active_dims]))