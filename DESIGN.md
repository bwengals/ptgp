# PTGP Design

A practical GP library for advanced users who need modeling flexibility, built on PyTensor's symbolic graph and rewrite system. Not a paper implementation zoo — a yeoman's GP library with SVGP as the primary workhorse model.

## Stack

- **PyTensor**: symbolic computation, graph rewrites, kernel matrix structure annotations (using PR #2032 `assumption-system` branch)
- **PyMC**: `pm.Model()` as prior container only — no PyMC samplers or inference routines
- **JAX + Optax**: JIT compilation and optimization via `get_jaxified_graph` (in `ptgp/inference/`)
- **Native PyTensor optimizers**: Adam/SGD via `pytensor.shared` variables (in `ptgp/optim/`) — no JAX dependency required
- **Kernels**: implemented natively in PTGP; long-term goal is for PyMC to depend on PTGP for kernels

## Models

| Model | Purpose |
|-------|---------|
| `Unapproximated` | Exact full GP |
| `VFE` | Variational Free Energy sparse GP |
| `SVGP` | Stochastic Variational GP — the workhorse |

All three are user facing. SVGP is the primary model for large datasets. VFE for cases that don't need minibatching. `Unapproximated` for small datasets.

## Prior / Inference Design

- `pm.Model()` is used **only** as a container for the prior graph
- `model.logp()` gives a symbolic PyTensor prior log-prob expression
- Two optimization paths:
  - **JAX-based** (`ptgp/inference/`): L-BFGS-B or Optax SGD via JAX compilation
  - **Native PyTensor** (`ptgp/optim/`): Adam/SGD using `pytensor.shared` variables — prediction automatically uses trained parameters
- Kernel and likelihood hyperparameters are PyMC RVs defined in `pm.Model` context

**User-facing pattern (native PyTensor):**
```python
import ptgp as pg

X_var = pt.matrix("X")
y_var = pt.vector("y")

with pm.Model() as model:
    ls    = pm.InverseGamma("ls", alpha=2.4, beta=1.5)
    eta   = pm.Exponential("eta", lam=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
    gp = pg.gp.Unapproximated(kernel=kernel, likelihood=pg.likelihoods.Gaussian(sigma=sigma))

# Compile training step — parameters stored as shared variables
train_step, shared_params, shared_extras = pg.optim.compile_training_step(
    pg.objectives.marginal_log_likelihood, gp, X_var, y_var, pm_model=model,
)

for i in range(500):
    loss = train_step(X_train, y_train)

# Compile prediction — reads same shared variables, no model reconstruction
X_new_var = pt.matrix("X_new")
predictn = pg.optim.compile_predict(gp, X_new_var, model, shared_params,
                                X_train=X_train, y_train=y_train)
mu, var = predictn(X_test)
```

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

Each objective is 1:1 with a model class and cannot be used interchangeably. They are standalone functions rather than methods because `compile_training_step` takes the objective as a first-class callable with a uniform `(model, X, y)` signature. This makes composition straightforward — e.g. wrapping the ELBO with minibatch scaling:

```python
def elbo_scaled(model, X, y):
    return pg.objectives.elbo(model, X, y, n_data=N)

pg.optim.compile_training_step(elbo_scaled, svgp, X_var, y_var, ...)
```

If objectives were methods, passing them to `compile_training_step` would require lambdas or wrappers.

## Core GP Math

Shared `base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False)` function (following GPflow) implements posterior conditional math for all three models. Linear algebra written naively, delegated to PyTensor's rewrite system.

## Inducing Variables

Class hierarchy following GPflow, enabling dispatch for inter-domain/multiscale/structured inducing points:

- `InducingVariables` — base class
- `Points(Z)` — standard real-space inducing points

## Likelihoods

- `Likelihood` base class with `variational_expectation(y, mu, var)`
- **Gaussian**: closed-form variational expectation
- **Non-Gaussian**: Gauss-Hermite quadrature (20 points default, matching GPflow) using `pm.logp(dist.dist(...), y)`
- Hyperparameters are PyMC RVs passed at construction

## Optimization (`ptgp/optim/`)

Native PyTensor training without JAX. Uses `pytensor.shared` variables so that trained parameters are automatically available for prediction without reconstructing the model.

### Architecture

- **`optimizers.py`**: Adam and SGD, each taking `(loss, params, **kwargs)` and returning `OrderedDict{shared_var: update_expr}`. Ported from `pymc.variational.updates`.
- **`training.py`**: `compile_training_step()` and `compile_predict()`.
  - `_make_shared_params()` creates `pytensor.shared` variables from PyMC's `initial_point()`
  - `_replace_graph()` substitutes PyMC RVs → value vars → shared vars using `graph_replace`
  - Optimizer updates are passed to `pytensor.function(..., updates=...)` so calling the function updates parameters in-place
  - `compile_predict()` builds a prediction function using the same shared variables — no model reconstruction needed

### SVGP extra variables

For SVGP, variational parameters (`q_mu`, `q_sqrt`) are not PyMC RVs. Pass them as `extra_vars` with `extra_init` to `compile_training_step`. They get their own shared variables and are optimized alongside the PyMC parameters.

### Known issue: recursion depth

PyTensor's assumption-system branch recursively traverses the graph to infer matrix properties. SVGP + Adam creates a large enough graph that this exceeds Python's default recursion limit. Use `sys.setrecursionlimit(50000)` before compiling SVGP models. The rewrite failures are caught gracefully and don't affect correctness.

## Prediction

- `predict()` returns symbolic PyTensor tensors
- With `ptgp/optim/`, use `compile_predict()` to get a callable that reads trained shared parameters

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
10. ~~`ptgp/optim/`~~ ✓

## Multi-Output GPs (planned)

Following GPflow's architecture. Data convention: `X (N, D)`, `Y (N, P)` where P is the number of outputs, L is the number of latent GPs.

### Multi-output kernels

Wrappers that produce per-output or per-latent covariance matrices:

- `SharedIndependent(kernel, output_dim)` — same kernel for all outputs. Returns `(P, N, N)`.
- `SeparateIndependent([k1, k2, ...])` — different kernel per output. Returns `(P, N, N)`.
- `LinearCoregionalization(kernels, W)` — L latent GPs mixed through a `(P, L)` weight matrix W. The key multi-output model for correlated outputs.

### Multi-output inducing variables

- `SharedIndependentInducingVariables(iv)` — same Z `(M, D)` for all latent GPs.
- `SeparateIndependentInducingVariables([iv1, iv2, ...])` — different Z per latent GP.

### Conditionals dispatch

A dispatch layer above `base_conditional` selects the right implementation based on (kernel type, inducing variable type):

- **SharedIndependent + SharedInducing**: call `base_conditional` once, broadcast across P outputs.
- **SeparateIndependent + SeparateInducing**: call `base_conditional` L times, one per latent GP.
- **LinearCoregionalization**: call `base_conditional` L times for latent GPs, then mix with `f = W @ g`.

The existing `base_conditional` handles single-output math and does not change.

### Variational parameter shapes

- `q_mu`: `(M, L)` — variational mean per latent GP.
- `q_sqrt`: `(L, M, M)` — Cholesky factor per latent GP.
- `gauss_kl` extends to sum L independent KL terms.

### Build order

1. `SharedIndependent` + `SeparateIndependent` kernels (independent outputs)
2. Corresponding inducing variable types
3. Dispatch layer above `base_conditional`
4. Extend `gauss_kl` to sum over L latent GPs
5. `LinearCoregionalization` for correlated outputs

## Future Directions

- Upstream `ptgp/linalg/` to PyTensor
- PyMC to depend on PTGP for GP kernels
- Toeplitz structured matrix support
- Sparse kernel support
- Natural/expectation variational parameterizations
- Inter-domain and multiscale inducing variable types
