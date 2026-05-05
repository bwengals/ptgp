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
- **Kernel annotations**: When a kernel evaluates `K(X, X)`, annotate with `pt.assume(K, symmetric=True, positive_definite=True)`. Cross-covariance `K(X, Y)` gets no annotation. (`pt.assume` is the public-facing name; ptgp patches it locally to also accept `positive=True` for scalar-positivity assumptions until upstream picks that up.)
- **Symbolic outputs**: All kernel, likelihood, and model methods return symbolic PyTensor tensors, not eagerly evaluated results.
- **Native kernels**: Kernels are implemented in PTGP, not reused from PyMC.

## Conventions

- **Function naming**: action first, then narrower qualifiers. `minimize_staged_vfe` not `staged_vfe_minimize`. `compile_scipy_objective` not `objective_scipy_compile`.
- **Scale parameter**: Use `eta` for kernel amplitude. Kernels are scaled by `eta**2` (e.g. `eta**2 * ExpQuad(ls=ls)`), so `eta` is always squared.
- **Lengthscale**: Use `ls` for lengthscale parameters.
- **Multiple parameters**: When there are multiple scale or lengthscale parameters (e.g. additive kernels), number them: `eta1`, `eta2`, `ls1`, `ls2`.
- **Name symbolic variables in complex examples**: Pass `name="..."` when constructing PyTensor variables (`pt.matrix("X")`, `pt.dmatrix("Z_var")`, etc.) in examples that involve per-group learning rates, per-group schedules, or multi-phase training loops. In those settings, symbolic vars become dict keys (e.g. in `param_groups`) and show up in error messages — error text falls back to `var.name or repr(var)`, and unnamed vars render as opaque addresses. Simple single-phase examples don't need this.
- **Prefix internal symbolic var names with `_`**: Any PyTensor variable named inside library code (e.g. `pt.vector("_theta")`, `pt.matrix("_X_opt")`) must start with an underscore. User-named variables never start with `_`, so the prefix distinguishes library-internal names from user names in error messages, graph dumps, and debugging. Example: `pt.matrix("_X", dtype="float64")` inside `greedy_variance_init`.
- **Error messages**: Keep them concise and include either a brief suggestion of the fix or the reason/contradiction that caused the error. One or two short sentences. The first sentence names the problem; the second names the reason or the fix. Example: `"Variables appear in both extra_vars and frozen_vars: [...]. They cannot be both trainable and frozen."`
- **Shape annotations on symbolic vars**: When defining `pt.matrix(...)`, `pt.vector(...)`, etc., pass `shape=(...)` for every dimension known at construction time. Use `None` only when the dimension genuinely varies between calls (typically the data axis `N`, which differs across training/eval/batches). Standard letters used throughout ptgp:
  - `N` — number of data points (almost always `None`)
  - `M` — number of inducing points (concrete int, known when `Z` / `init_variational_params(M)` is constructed)
  - `D` — input dimension / columns of `X` (concrete int, known when the kernel is constructed)
  - `K` — number of outputs / columns of `Y` (concrete int, when multi-output is added)

  Examples:
  ```python
  X    = pt.matrix("X",    shape=(None, D))
  y    = pt.vector("y",    shape=(None,))
  Z    = pt.matrix("Z",    shape=(M, D))
  q_mu = pt.vector("q_mu", shape=(M,))
  ```

  Why: shape info lets PyTensor's rewrite system specialize (e.g., the future Woodbury rewrite gates on `M < N`), catches shape mismatches at compile time, and documents axis intent. Even single-axis annotations (`shape=(None, 1)` for 1-D data) are worth it. Applies to library code, tests, examples, and notebooks alike.

## Testing

- GPJax is the reference implementation for correctness tests.
- GPJax uses float32; PTGP uses float64. Use `atol=1e-5` for cross-library comparisons.
- Test pattern: set up identical inputs, evaluate both libraries, compare with `np.testing.assert_allclose`.

## Notebooks

- Jupyter notebooks are stored as `.ipynb` files and edited directly. No Jupytext pairing or markdown mirror.

## Environment

- **All code and notebooks run on the `ptgp` conda env** (Python 3.14, at `/Users/bill/miniconda3/envs/ptgp`). Never use the system python or a different env — pinned versions and the custom PyTensor branch live only here.
- Run tests: `/Users/bill/miniconda3/envs/ptgp/bin/python -m pytest tests/`
- Run a script: `/Users/bill/miniconda3/envs/ptgp/bin/python <script>`
- Execute a notebook in place: `/Users/bill/miniconda3/envs/ptgp/bin/python -m jupyter nbconvert --to notebook --execute --inplace <path> --ExecutePreprocessor.kernel_name=ptgp` (the `ptgp` kernel is registered globally and points at this env)
- PyTensor is installed from the `assumption-system` branch (PR #2032) which provides `pt.assume` and the assumption-tag system.

## Commits

Prefer one-sentence commit messages for smaller and simpler changes. Use a one-sentence subject + multi-line body only for more complex changes that genuinely warrant explanation (multi-file refactors, non-obvious design decisions, behavior changes that need context).

After finishing a plan or feature, prompt the user whether to `git add` and commit the recent changes. Don't auto-commit; wait for explicit confirmation before staging or committing.

## Skills (`.claude/skills/`)

The repo ships a `ptgp-vfe` skill at `.claude/skills/ptgp-vfe/`. When changing any of:

- `ptgp/objectives.py` — especially `VFEDiagnostics`, `CollapsedELBOTerms`, `vfe_diagnostics`, `collapsed_elbo`, `dpp_regularizer`
- `ptgp/optim/training.py` — especially `compile_scipy_objective`, `compile_scipy_diagnostics`, `tracked_minimize`, `minimize_staged_vfe`
- `ptgp/inducing.py` — especially `GreedyVarianceDiagnostics`, `greedy_variance_init`, `kmeans_init` dedup logic
- `ptgp/utils.py` — `check_init`, `get_initial_params`

…check whether the skill needs updating: `reference/api.md` for signatures, `reference/interpretation.md` for renamed/added namedtuple fields, `pitfalls/*.md` for any references to renamed fields. This keeps the in-repo skill current with the code.
