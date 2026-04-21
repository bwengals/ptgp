# VFF + Analytic Variational Expectations — Design

Date: 2026-04-21
Status: Revised after Codex adversarial review (see "Review response" below).

Two scale-unlock features for PTGP's SVGP path, planned as two independent PRs. PR 1 adds Variational Fourier Features as a new inducing class. PR 2 hardens and extends the existing Poisson analytic-VE path.

## Review response

The first version of this doc claimed PR 2 would "add" an analytic variational expectation for Poisson with log link and proposed a `Poisson(link='exp'|'softplus')` API. Reading `ptgp/likelihoods/poisson.py` during the revision made it clear that:

1. `Poisson` already defaults to `invlink=pt.exp`.
2. `Poisson.variational_expectation` already dispatches to the exact closed form `y·μ − exp(μ + σ²/2) − pt.gammaln(y+1)` when `invlink is pt.exp`, and falls back to quadrature otherwise.
3. `tests/likelihoods/test_poisson.py` already checks the closed form matches 50-point Gauss-Hermite at `atol=1e-6`.

The proposed `link='exp'|'softplus'` string API would have been a *breaking* API change for anyone passing a custom callable, and the "fast path activates by default" claim was true only by coincidence (because the existing default is already `pt.exp`). PR 2 is therefore rescoped to **hardening work** around the existing analytic path (end-to-end SVGP coverage, graph-level regression assertions per Finding 4), and the API change is dropped.

PR 1 is kept but tightened: VFF now specifies a concrete domain-validation policy (Finding 2), a typed structured-Kuu contract distinct from the dense `K_uu` path (Finding 3), and CI-asserted algorithmic-path tests (Finding 4).

## Context

PTGP targets applied power users running SVGP at N = 10⁵–10⁷. Two bottlenecks dominate in practice:

1. **Inducing-point capacity.** `Points(Z)` at M=500 captures less of a wiggly 1D signal than a Fourier basis at the same M.
2. **Non-Gaussian likelihood cost.** The default Gauss-Hermite quadrature in `Likelihood.variational_expectation` evaluates `log p(y|f)` 20–50 times per datapoint per ELBO call. Poisson-log dodges this *if and only if* the fast path actually runs in the compiled graph.

---

# PR 1 — Variational Fourier Features (1D, Matérn)

Hensman, Durrande, Solin 2017. A new `InducingVariables` subclass where inducing "features" are Fourier-basis projections on an interval. For Matérn kernels the prior covariance `Kuu` is diagonal plus low-rank; combining that with Woodbury gives O(M) linear solves.

## Scope

**In.** 1D inputs. Matérn ½, 3/2, 5/2. SVGP and VFE wiring. Domain validation. Graph-level regression tests.

**Out, this PR.** Tensor-product 2D/3D extension. ExpQuad. Spherical harmonic features. Kronecker interactions.

Rationale: per-kernel math is already nontrivial. 1D Matérn is the killer feature on its own.

## Architecture

Two extension points on existing abstractions, no changes to SVGP/VFE model bodies.

### Typed structured-Kuu contract (Finding 3)

The first revision conflated "dense matrix returned by `K_uu`" with "structured container returned by `K_uu`," producing an ambiguous contract where the same method could return either. Fixed by splitting into two named methods with disjoint return types:

- `InducingVariables.K_uu(kernel) -> pt.TensorVariable`
  Dense M×M prior covariance. Every subclass implements this honestly (including VFF, which materialises the dense `Kuu` only for tests and for fallback consumers).
- `InducingVariables.K_uf(kernel, X) -> pt.TensorVariable`
  Dense M×N cross-covariance.
- `InducingVariables.structured_Kuu(kernel) -> StructuredMatrix | None`
  Optional. `Points` returns `None`. `FourierFeatures1D` returns a typed `DiagPlusLowRank(diag, U)` dataclass.
- `InducingVariables.solve_Kuu(kernel, rhs) -> pt.TensorVariable`
  Optional. Default implementation calls `pt.linalg.solve(self.K_uu(kernel), rhs)`. `FourierFeatures1D` overrides with Woodbury using its `structured_Kuu`.

`Points(Z)` keeps current semantics: dense `K_uu`, `K_uf`, no `structured_Kuu`, default dense `solve_Kuu`.

`ptgp/conditionals.py` has exactly one explicit dispatch point: the single call site that inverts `Kuu` calls `inducing.solve_Kuu(kernel, rhs)`. No type-switching on the return of `K_uu`. This means even a buggy consumer that calls `K_uu` directly on a VFF object gets a dense matrix (correct result, slow) rather than a shape-incompatible structured tuple — no silent shape drift.

### FourierFeatures1D closed forms

```
φ_0(x)       = 1 / sqrt(b - a)                (DC)
φ_{2k-1}(x)  = sqrt(2/(b-a)) * cos(ω_k (x - a))   for k = 1..M
φ_{2k}(x)    = sqrt(2/(b-a)) * sin(ω_k (x - a))   for k = 1..M
ω_k          = k * π / (b - a)
```

`num_inducing = 2*num_frequencies + 1`. `K_uf(kernel, X)` evaluates `φ_m(X)` (no kernel dependence — reproducing property gives exact inducing-feature projections for Matérn). `structured_Kuu(kernel)` returns the diagonal-plus-low-rank decomposition of the RKHS inner product `<φ_m, φ_n>_H` as worked out in the paper (kernel-specific coefficients for each Matérn order).

## API

```python
import ptgp as pg

kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
likelihood = pg.likelihoods.Gaussian(sigma=sigma)
inducing = pg.inducing.FourierFeatures1D(a=0.0, b=10.0, num_frequencies=64)

gp = pg.gp.SVGP(kernel=kernel, likelihood=likelihood, inducing=inducing)
# compile_training_step, loop, compile_predict unchanged
```

Helper `FourierFeatures1D.from_data(X, num_frequencies, buffer=0.1)` picks `(a, b) = (X.min() − buffer·range, X.max() + buffer·range)`.

## Domain validation (Finding 2)

The Fourier basis is defined on `[a, b]`. Outside this interval the basis is not zero — it's periodic — so extrapolation silently returns a periodic continuation of the in-domain fit. This is almost never what the user wants and is the most likely source of "it looks fine but the prediction is wrong" bugs.

Policy:

- **Construction.** `FourierFeatures1D(a, b, num_frequencies)` validates `a < b` and `num_frequencies >= 1`. Error message names the violating argument.
- **Training.** The first time the compiled training graph runs, the training-data check `pt.and_(X.min() >= a, X.max() <= b)` is embedded via `pt.check` with a message `"training X has points outside the VFF domain [a, b]; widen the domain or use FourierFeatures1D.from_data(X, ...)"`. Cost: one reduction per training call; negligible.
- **Prediction.** `compile_predict` emits the same `pt.check`. Default behaviour: out-of-domain predictions raise. An opt-in `allow_extrapolation=True` kwarg on the inducing class skips the check and documents (in the docstring) that the prediction is a periodic continuation.
- **Tests.**
  - `test_fourier_rejects_out_of_domain_train` — construct with `(a, b) = (0, 1)`, call training with `X.max() = 1.5`, expect a clear error mentioning the domain.
  - `test_fourier_rejects_out_of_domain_predict` — same for prediction.
  - `test_fourier_extrapolation_opt_in` — `allow_extrapolation=True` passes and produces the documented periodic-continuation values.

## Tests

### Correctness

- **Convergence to exact GP.** VFF SVGP ELBO and predictions converge to exact GP as `num_frequencies → large` with domain covering data. N=500 Matérn-½/3/2/5/2 draws. `atol=1e-3`.
- **GPflow cross-check (opt-in).** If GPflow's VFF is present in a sibling checkout, cross-check `Kuu`, `Kuf`, posterior mean at matched hyperparameters, `atol=1e-5`. Skipped if GPflow absent — no hard CI dep.
- **Woodbury correctness.** `FourierFeatures1D.solve_Kuu(rhs)` matches `pt.linalg.solve(K_uu_dense, rhs)` at `atol=1e-6`.

### Algorithmic-path regression (Finding 4)

- **Dense solve is absent from the VFF compiled graph.** Build and compile the SVGP training step with `FourierFeatures1D` and with `Points` for the same kernel. Walk the compiled graph (`f.maker.fgraph.toposort()`) and assert no `Solve`/`MatrixInverse`/`Cholesky` Op is applied to a dense M×M tensor in the VFF graph. A tolerant version: assert that the total count of such Ops is strictly lower in the VFF graph than in the `Points` graph and is bounded by a small constant. This test is the canary for "we accidentally fell back to dense."
- **Prediction graph regression.** Same check for `compile_predict`.

### Speed sanity (notebook, not asserted)

Single benchmark: SVGP training-step time on N=50k, M=200 VFF vs M=200 Points, 100 steps each. Kept in a notebook for visibility; not a CI gate.

## Risks

- **Numerical conditioning at high M near domain edges.** Well-known VFF failure mode. Mitigations: `from_data` adds buffer; high-M regression tests flag when `solve_Kuu` residuals exceed `1e-4`.
- **Structured container escaping.** If `structured_Kuu` leaks into a context that expects a tensor, the typed return (`DiagPlusLowRank` dataclass, not a tuple) surfaces as a clear `TypeError` rather than a mysterious shape failure. The conditional never calls `structured_Kuu` directly — only `solve_Kuu`.
- **Recursion-limit gotcha.** Same `sys.setrecursionlimit(50000)` as current SVGP; verify during implementation.

---

# PR 2 — Harden existing Poisson analytic VE

## Current state (ground truth from code)

`ptgp/likelihoods/poisson.py`:

- `Poisson(invlink=None, n_points=20)` defaults `invlink` to `pt.exp`.
- `variational_expectation(y, mu, var)` checks `self.invlink is pt.exp`; if true, returns the exact closed form `y·μ − exp(μ + var/2) − pt.gammaln(y+1)`; otherwise falls back to `Likelihood._gauss_hermite`.
- `tests/likelihoods/test_poisson.py::test_closed_form_matches_quadrature` asserts the closed form matches 50-point Gauss-Hermite at `atol=1e-6`.

So the feature itself is already shipped. What's missing is exactly what Codex flagged: no assertion that the fast path actually runs in the compiled SVGP graph, no end-to-end SVGP Poisson test, and no gradient-correctness test.

## Scope

- Add SVGP end-to-end test for Poisson-log: `Poisson()` default SVGP on synthetic count data converges to expected posterior mean at `atol=1e-4`, matches the quadrature version.
- Add gradient-correctness test: JAX autograd through the analytic VE matches finite-differences w.r.t. `mu` and `var`.
- Add algorithmic-path regression: compile the SVGP training step for `Poisson()` (fast path) and for `Poisson(invlink=lambda f: pt.softplus(f))` (quadrature). Assert the fast-path graph contains no Gauss-Hermite nodes and the slow-path graph does. Canary for "someone broke identity-dispatch."
- **No API change.** The `invlink` callable API stays. The `link='exp'|'softplus'` string proposal from the first revision is dropped — it would have been a breaking change for anyone passing a custom `invlink`, and it conflated "link choice" with "did the fast path activate."

## Architecture

No code changes. Test-only PR.

## Tests

- **End-to-end SVGP Poisson.** Synthetic N=2000 counts with known rate; SVGP trains, posterior mean converges to ground truth at `atol=0.1`, ELBO matches quadrature-fallback ELBO at `atol=1e-4`.
- **Gradient correctness.** `jax.grad(ve)(mu, var)` vs finite differences at a grid of `(y, μ, σ²)`; `atol=1e-5`.
- **Algorithmic-path regression.** Walk the compiled training graph for `Poisson()` and assert the `_gauss_hermite` helper's quadrature-weight constants (`np.polynomial.hermite.hermgauss(20)` / `(50)`) are *not* present. Positive control: same check on `Poisson(invlink=pt.nnet.softplus)` *is* present. Mirror of the VFF graph-regression test.

## Risks

- **Fast-path silently broken by refactor.** This is exactly what the algorithmic-path regression catches.
- **`log(y!)` overflow on very large counts.** Already handled: `pt.gammaln(y + 1)` in the current code.

---

## PR ordering

Independent. Suggested: **PR 2 first** (test-only, small, warms up the graph-introspection utility that PR 1 then reuses), then PR 1.

## Shared infrastructure

The "is this Op in the compiled graph" helper needed for both algorithmic-path tests is the same utility in both PRs. Land it with PR 2 under `tests/_graph_helpers.py`; PR 1 imports it.
