# PTGP — Instructions for Claude

## What this is

A Gaussian process library built on PyTensor's symbolic graph and rewrite system. See `DESIGN.md` for full architecture.

## Stage

Prototype. Keep code and tests brief. The goal is a working prototype, not completeness.

## Reference code

Read source code in `comparison_libraries/` (GPJax, GPflow, GPyTorch, CoLA, linear_operator, PyTensor, PyMC). Implementations of kernels, likelihoods, GP approximations, and variational inference methods are based on these. GPJax is the primary reference for writing tests. Read the DESIGN.md file before beginning.

## Key constraints

- **No PyMC inference**: Never use `pm.sample()`, `pm.find_MAP()`, or any PyMC inference routine. `pm.Model()` is only a prior container.
- **Naive linear algebra**: Write `pt.linalg.inv(K)`, `pt.linalg.slogdet(K)` directly. Do not manually write Cholesky + solve_triangular chains. Let PyTensor's rewrite system choose algorithms.
- **Kernel annotations**: When a kernel evaluates `K(X, X)`, annotate with `pt.specify_assumptions(K, symmetric=True, positive_definite=True)`. Cross-covariance `K(X, Y)` gets no annotation.
- **Symbolic outputs**: All kernel, likelihood, and model methods return symbolic PyTensor tensors, not eagerly evaluated results.
- **Native kernels**: Kernels are implemented in PTGP, not reused from PyMC.

## Conventions

- **Scale parameter**: Use `eta` for kernel amplitude. Kernels are scaled by `eta**2` (e.g. `eta**2 * ExpQuad(ls=ls)`), so `eta` is always squared.
- **Lengthscale**: Use `ls` for lengthscale parameters.
- **Multiple parameters**: When there are multiple scale or lengthscale parameters (e.g. additive kernels), number them: `eta1`, `eta2`, `ls1`, `ls2`.
- **Name symbolic variables in complex examples**: Pass `name="..."` when constructing PyTensor variables (`pt.matrix("X")`, `pt.dmatrix("Z_var")`, etc.) in examples that involve per-group learning rates, per-group schedules, or multi-phase training loops. In those settings, symbolic vars become dict keys (e.g. in `param_groups`) and show up in error messages — error text falls back to `var.name or repr(var)`, and unnamed vars render as opaque addresses. Simple single-phase examples don't need this.

## Testing

- GPJax is the reference implementation for correctness tests.
- GPJax uses float32; PTGP uses float64. Use `atol=1e-5` for cross-library comparisons.
- Test pattern: set up identical inputs, evaluate both libraries, compare with `np.testing.assert_allclose`.

## Notebooks

- Jupyter notebooks are stored as `.ipynb` files and edited directly. No Jupytext pairing or markdown mirror.

## Environment

- Conda env: `ptgp` (Python 3.14, at `/Users/bill/miniconda3/envs/ptgp`)
- Run tests: `/Users/bill/miniconda3/envs/ptgp/bin/python -m pytest tests/`
- PyTensor is installed from the `assumption-system` branch (PR #2032) which provides `pt.specify_assumptions`
