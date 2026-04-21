# PTGP

A Gaussian process library built on PyTensor's symbolic graph and rewrite system.

PTGP writes GP math directly and simply — `pt.linalg.inv(K)`, `pt.linalg.slogdet(K)` — and lets PyTensor's rewrite system choose efficient algorithms based on declared matrix structure (symmetric, positive definite). This is fundamentally different from libraries like GPyTorch/CoLA that select algorithms at runtime.

## Models

| Model | Description |
|-------|-------------|
| `gp.Unapproximated` | Exact full Gaussian process |
| `gp.VFE` | Variational Free Energy sparse GP (Titsias' collapsed bound) |
| `gp.SVGP` | Stochastic Variational GP — the workhorse for large datasets |

## Quick start

```python
import numpy as np
import pytensor.tensor as pt
import ptgp as pg

# Data
X = np.random.randn(100, 1)
y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)

# Symbolic parameters (unconstrained)
log_ls = pt.scalar("log_ls")
log_eta = pt.scalar("log_eta")
log_sigma = pt.scalar("log_sigma")

# Build model
kernel = pt.exp(log_eta)**2 * pg.kernels.Matern52(input_dim=1, ls=pt.exp(log_ls))
gp = pg.gp.Unapproximated(kernel=kernel, likelihood=pg.likelihoods.Gaussian(sigma=pt.exp(log_sigma)))

# Optimize hyperparameters
opt_values, result = pg.inference.fit_bfgs(
    pg.objectives.marginal_log_likelihood, gp, X, y,
    params=[log_ls, log_eta, log_sigma],
    init_values=[0.0, 0.0, -1.0],
)
```

## Stack

- **PyTensor**: symbolic computation and graph rewrites (using the `assumption-system` branch, PR #2032)
- **JAX + Optax**: JIT compilation and optimization via `pytensor.function(..., mode='JAX')`
- **PyMC** (planned): `pm.Model()` as a prior container for hyperparameters

## Kernels

Kernels evaluate on matrix pairs and return annotated PyTensor tensors:

- `ExpQuad`, `Matern52`, `Matern32`, `Matern12` — stationary kernels
- `RandomWalk` — nonstationary kernel
- Composition via `+` and `*` operators
- Scaling via `eta**2 * Kernel(ls=ls)`

## Likelihoods

- `Gaussian` — closed-form variational expectations
- `Bernoulli` — probit link with Gauss-Hermite quadrature
- `StudentT`, `Poisson`, `NegativeBinomial` — quadrature-based
- Configurable link functions

## Install

```bash
pip install -e .
```

Requires the `assumption-system` branch of PyTensor:

```bash
pip install git+https://github.com/jessegrabowski/pytensor@assumption-system
```
