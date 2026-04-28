# PTGP

A Gaussian process library built on PyTensor's symbolic graph and rewrite system.

PTGP writes GP math directly and simply — `pt.linalg.inv(K)`, `pt.linalg.slogdet(K)` — and lets PyTensor's rewrite system choose efficient algorithms based on declared matrix structure (`pt.assume(K, symmetric=True, positive_definite=True)`). All three of `Unapproximated`, `VFE`, and `SVGP` compile their full forward+gradient training step down to the *cubic-factorisation floor* — one Cholesky/step for the exact GP and SVGP, two for VFE. Pinned by `tests/test_cubic_floor.py`. This is fundamentally different from libraries like GPyTorch/CoLA that select algorithms at runtime.

## Models

| Model | Description |
|-------|-------------|
| `gp.Unapproximated` | Exact full Gaussian process |
| `gp.VFE` | Variational Free Energy sparse GP (Titsias' collapsed bound) |
| `gp.SVGP` | Stochastic Variational GP — the workhorse for large datasets |

## Quick start

Hyperparameters are declared as PyMC RVs inside a `pm.Model()` block; the model is used only as a prior container (no PyMC inference). Training runs in native PyTensor with `pytensor.shared` variables, so prediction reads the trained parameters automatically.

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.optimize
import ptgp as pg

X = np.random.randn(100, 1)
y = np.sin(X.ravel()) + 0.1 * np.random.randn(100)

with pm.Model() as model:
    ls    = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
    eta   = pm.Exponential("eta", lam=1.0)
    sigma = pm.Exponential("sigma", lam=1.0)

    kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
    gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

# Annotate static shapes wherever known — helps PyTensor's rewrite system
# specialize and (eventually) lets shape-aware rewrites like Woodbury fire.
X_var = pt.matrix("X", shape=(None, 1))
y_var = pt.vector("y", shape=(None,))

# Train with L-BFGS-B (scipy owns the parameter vector; no gradient plumbing needed).
fun, theta0, unpack_to_shared, shared_params, _ = pg.optim.compile_scipy_objective(
    pg.objectives.marginal_log_likelihood, gp, X_var, y_var, model=model,
)
result = scipy.optimize.minimize(fun, theta0, args=(X, y), jac=True, method="L-BFGS-B")
unpack_to_shared(result.x)  # writes trained values into the shared vars used by predict

X_new_var = pt.matrix("X_new", shape=(None, 1))
predict_fn = pg.optim.compile_predict(gp, X_new_var, model, shared_params,
                                       X_train=X, y_train=y)
mean, var = predict_fn(np.linspace(-3, 3, 50)[:, None])
```

Training uses MAP by default — the PyMC joint log-prior is added to the objective. Pass `include_prior=False` for pure MLE/ELBO. For SVGP's minibatched ELBO, use `compile_training_step` with Adam/SGD instead; SVGP's variational parameters are constructed via `pg.gp.init_variational_params(M)` and threaded through as `extra_vars` (see the demo notebook for the full pattern).

For the full treatment of `Unapproximated`, `VFE`, and `SVGP` (including minibatched training, staged variational init, and both L-BFGS-B and Adam paths), see [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

## Stack

- **PyTensor** (`assumption-system` branch, PR #2032): symbolic computation and graph rewrites driven by declared matrix structure (`pt.assume`). PTGP installs additional rewrites on top — see `REWRITE_ANALYSIS.md` for the full diagnosis of how the rewrite suite reaches the cubic-factorisation floor.
- **PyMC**: `pm.Model()` as a prior container for hyperparameters — no PyMC samplers or inference routines
- **Native PyTensor optimizers** (`ptgp.optim`): L-BFGS-B (via scipy) and Adam / SGD with `pytensor.shared` parameters — no JAX required

## Kernels

Kernels evaluate on matrix pairs and return annotated PyTensor tensors:

- **Stationary:** `ExpQuad`, `Matern52`, `Matern32`, `Matern12`
- **Nonstationary:** `RandomWalk`, `Gibbs`, `WarpedInput`
- **Categorical:** `Overlap` (Hamming-style), `LowRankCategorical` (ICM-form)
- Composition via `+` and `*` (`SumKernel`, `ProductKernel`)
- `active_dims` for selecting input dimensions
- Scaling via `eta**2 * Kernel(ls=ls)`

## Likelihoods

- `Gaussian` — closed-form variational expectations
- `Bernoulli`, `StudentT`, `Poisson`, `NegativeBinomial` — Gauss–Hermite quadrature for the variational expectation (20 points default, matching GPflow)
- Hyperparameters are PyMC RVs passed at construction

## Install

```bash
pip install -e .
```

Requires the `assumption-system` branch of PyTensor:

```bash
pip install git+https://github.com/jessegrabowski/pytensor@assumption-system
```
