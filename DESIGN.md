# PTGP Design

A practical GP library for advanced users who need modeling flexibility, built on PyTensor's symbolic graph and rewrite system. Not a paper implementation zoo — a yeoman's GP library with SVGP as the primary workhorse model.

## Stack

- **PyTensor**: symbolic computation, graph rewrites, kernel matrix structure annotations (using PR #2032 `assumption-system` branch)
- **PyMC**: `pm.Model()` as prior container only — no PyMC samplers or inference routines
- **JAX + Optax**: JIT compilation and optimization via `get_jaxified_graph` — long-term goal is native PyTensor training routines
- **Kernels**: implemented natively in PTGP; long-term goal is for PyMC to depend on PTGP for kernels

## Models

| Model | Purpose |
|-------|---------|
| `GP` | Exact full GP |
| `VFE` | Variational Free Energy sparse GP |
| `SVGP` | Stochastic Variational GP — the workhorse |

All three are user facing. SVGP is the primary model for large datasets. VFE for cases that don't need minibatching. GP for small datasets.

## Prior / Inference Design

- `pm.Model()` is used **only** as a container for the prior graph
- `model.logp()` gives a symbolic PyTensor prior log-prob expression
- PTGP compiles this to JAX and optimizes with L-BFGS-B or Optax SGD
- Kernel and likelihood hyperparameters are PyMC RVs defined in `pm.Model` context

**User-facing pattern:**
```python
with pm.Model() as model:
    ls    = pm.LogNormal("ls", mu=0, sigma=1)
    eta   = pm.HalfNormal("eta", sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    cov        = eta**2 * ptgp.kernels.ExpQuad(1, ls=ls)
    mean       = ptgp.mean.Zero()
    likelihood = ptgp.likelihoods.Gaussian(sigma)

    z = ptgp.InducingPoints(Z_init)
    svgp = ptgp.SVGP(mean, cov, likelihood, z, whiten=True)

gp.fit(X_train, y_train, method="sgd", optimizer=optax.adam(1e-3))
mu, var = gp.predict(X_new)
```

## Data Handling

Reuses PyMC's `pm.Data` for data containers, allowing data to be swapped for minibatching.

## Kernel Design

Kernels evaluate on **matrix pairs**, returning annotated PyTensor tensors (not GPJax's scalar-pair-with-vmap approach).

- `__call__(X, Y=None)` returns a PyTensor tensor
- `K(X, X)` annotated as `symmetric=True`, `positive_definite=True` via `pt.specify_assumptions`
- `K(X, Y)` returned unannotated
- Composition via `+` and `*` operators (`SumKernel`, `ProductKernel`)
- `active_dims` for selecting input dimensions
- **Kronecker structure**: explicit user declaration via `KroneckerKernel` wrapper
- **Sparse kernels**: planned pending PyTensor sparse matrix support

## Objectives

Standalone functions, not methods on model classes (follows GPJax's pattern):

- `marginal_log_likelihood(gp, X, y)` — exact GP
- `elbo(svgp, X_batch, y_batch)` — SVGP (scales by `n_data / batch_size`)
- `collapsed_elbo(vfe, X, y)` — VFE/SGPR (Titsias' bound)

## Core GP Math

Shared `base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False)` function (following GPflow) implements posterior conditional math for all three models. Linear algebra written naively, delegated to PyTensor's rewrite system.

## Inducing Variables

Class hierarchy following GPflow, enabling dispatch for inter-domain/multiscale/structured inducing points:

- `InducingVariables` — base class
- `InducingPoints(Z)` — standard real-space inducing points

## Likelihoods

- `Likelihood` base class with `variational_expectation(y, mu, var)`
- **Gaussian**: closed-form variational expectation
- **Non-Gaussian**: Gauss-Hermite quadrature (20 points default, matching GPflow) using `pm.logp(dist.dist(...), y)`
- Hyperparameters are PyMC RVs passed at construction

## Prediction

- `predict()` returns symbolic PyTensor tensors
- GP is stateful — stores training data after `fit()`

## SVGP Variational Parameterization

- **Whitened** (default): `u = Lz v`, prior on `v` is `N(0, I)`. KL simplifies to `KL[N(m, S) || N(0, I)]`.
- **Unwhitened**: `q(u) = N(m, S)` directly. KL is `KL[N(m, S) || N(0, Kzz)]`.

Natural/expectation parameterizations out of scope for now.

## KL Divergence

Standalone `gauss_kl(q_mu, q_sqrt, K=None)`:
- `K` provided: unwhitened KL
- `K is None`: whitened KL (prior is identity)

## Linear Algebra Style

Write GP math directly and simply. PyTensor's rewrite system selects efficient implementations:
- `pt.linalg.inv(K)` not manual Cholesky solves
- `pt.linalg.slogdet(K)` not manual log-det chains
- Declare structure via annotations, let the compiler choose algorithms

This is fundamentally different from GPyTorch/CoLA/linear_operator which choose algorithms at runtime.

## LinearOperator (`ptgp/linalg/`)

Custom PyTensor Ops for lazy kernel evaluation and iterative solvers. Lives in PTGP initially, planned for upstream to PyTensor.

Enables scaling via CG-based solves, lazy kernel matrix-vector products, and Lanczos-based log-determinant estimation. Deferred until after SVGP works with dense matrices.

## Build Order

1. ~~Project layout~~ ✓
2. ~~`ptgp/kernels/`~~ ✓
3. ~~`ptgp/likelihoods/`~~ ✓
4. ~~`ptgp/objectives.py`~~ ✓
5. ~~`ptgp/conditionals.py`~~ ✓
6. ~~`ptgp/gp.py`~~ ✓
7. ~~`ptgp/vfe.py`~~ ✓
8. `ptgp/linalg/` (deferred — algorithm selection depends on matrix size, which is symbolic at graph-build time; needs design work)
9. ~~`ptgp/svgp.py`~~ ✓

## Future Directions

- Upstream `ptgp/linalg/` to PyTensor
- PyMC to depend on PTGP for GP kernels
- Native PyTensor training routines (replace JAX/Optax)
- Toeplitz structured matrix support
- Sparse kernel support
- Natural/expectation variational parameterizations
- Inter-domain and multiscale inducing variable types
