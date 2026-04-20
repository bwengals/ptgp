import numpy as np


class Kernel:
    """Base class for all PTGP kernels.

    Subclasses must implement ``__call__(self, X, Y=None)``.  When called with
    a single argument (Y is None) the kernel matrix K(X, X) is returned and
    annotated as symmetric and positive-definite via PyTensor's assumption
    system.  When called with two arguments K(X, Y) is returned unannotated.

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    active_dims : sequence of int, optional
        Columns of ``X`` this kernel operates on. Defaults to all columns.
    """

    def __init__(self, input_dim, active_dims=None):
        """Validate and store ``input_dim`` and ``active_dims``."""
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.asarray(active_dims, dtype=int)
            if self.active_dims.max() >= input_dim:
                raise ValueError(
                    f"active_dims contains index {int(self.active_dims.max())}, "
                    f"but input_dim is {input_dim}"
                )

    def __call__(self, X, Y=None):
        """Return the kernel matrix K(X, Y) — or K(X, X) if Y is None."""
        raise NotImplementedError

    def __add__(self, other):
        """Return a SumKernel combining self and other."""
        from ptgp.kernels.combination import SumKernel

        return SumKernel(self, other)

    def __mul__(self, other):
        """Return a ProductKernel combining self and other."""
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)

    def __rmul__(self, other):
        """Return a ProductKernel; supports ``scalar * kernel``."""
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)