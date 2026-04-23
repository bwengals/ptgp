from ptgp import (
    gp,
    inducing,
    kernels,
    likelihoods,
    mean,
    objectives,
    optim,
)

__all__ = [
    "gp",
    "inducing",
    "kernels",
    "likelihoods",
    "mean",
    "objectives",
    "optim",
]


try:
    from ptgp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"