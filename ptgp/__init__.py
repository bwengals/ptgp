from ptgp import (
    gp,
    inducing,
    kernels,
    likelihoods,
    mean,
    objectives,
    optim,
    rewrites,  # noqa: F401  (registers PyTensor rewrites at import)
    utils,
)

__all__ = [
    "gp",
    "inducing",
    "kernels",
    "likelihoods",
    "mean",
    "objectives",
    "optim",
    "utils",
]


try:
    from ptgp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"