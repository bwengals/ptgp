from ptgp.gp import GP
from ptgp.inducing_variables import InducingPoints, InducingVariables
from ptgp.inference import fit_bfgs, fit_model, make_training_step, make_training_step_model
from ptgp.kernels import (
    ExpQuad,
    Gibbs,
    Kernel,
    Matern12,
    Matern32,
    Matern52,
    ProductKernel,
    RandomWalk,
    SumKernel,
    WarpedInput,
)
from ptgp.likelihoods import (
    Bernoulli,
    Gaussian,
    Likelihood,
    NegativeBinomial,
    Poisson,
    StudentT,
)
from ptgp.mean import Constant, Linear, Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood
from ptgp.optim import adam, compile_predict, compile_training_step, get_trained_params, sgd
from ptgp.svgp import SVGP
from ptgp.vfe import VFE

__all__ = [
    # Models
    "GP",
    "VFE",
    "SVGP",
    # Inducing variables
    "InducingPoints",
    "InducingVariables",
    # Mean functions
    "Zero",
    "Constant",
    "Linear",
    # Kernels
    "Kernel",
    "ExpQuad",
    "Matern52",
    "Matern32",
    "Matern12",
    "Gibbs",
    "RandomWalk",
    "WarpedInput",
    "SumKernel",
    "ProductKernel",
    # Likelihoods
    "Likelihood",
    "Gaussian",
    "Bernoulli",
    "StudentT",
    "Poisson",
    "NegativeBinomial",
    # Objectives
    "marginal_log_likelihood",
    "elbo",
    "collapsed_elbo",
    # Inference (JAX-based)
    "fit_bfgs",
    "fit_model",
    "make_training_step",
    "make_training_step_model",
    # Optim (native PyTensor)
    "adam",
    "sgd",
    "compile_training_step",
    "compile_predict",
    "get_trained_params",
]


try:
    from ptgp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
