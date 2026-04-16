class Kernel:
    """Base class for all PTGP kernels.

    Subclasses must implement ``__call__(self, X, Y=None)``.  When called with
    a single argument (Y is None) the kernel matrix K(X, X) is returned and
    annotated as symmetric and positive-definite via PyTensor's assumption
    system.  When called with two arguments K(X, Y) is returned unannotated.
    """

    def __call__(self, X, Y=None):
        raise NotImplementedError

    def __add__(self, other):
        from ptgp.kernels.combination import SumKernel

        return SumKernel(self, other)

    def __mul__(self, other):
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)

    def __rmul__(self, other):
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)
