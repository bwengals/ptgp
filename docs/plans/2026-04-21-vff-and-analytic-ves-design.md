# VFF + Analytic Variational Expectations — Design

Date: 2026-04-21
Status: Approved, ready for implementation planning.

Two independent scale-unlock features for PTGP, shipped as two separate PRs to keep reviews small. Both target the SVGP path on big-N, unstructured data — PR 1 by compressing inducing capacity, PR 2 by removing per-datapoint quadrature for the most common count-regression likelihood.

## Context

PTGP targets applied power users running SVGP on N in the 10⁵–10⁷ range. Two bottlenecks dominate in practice:

1. Inducing-point capacity. Real-space `Points(Z)` at M=500 captures less of a wiggly 1D signal than a Fourier basis with the same M.
2. Non-Gaussian likelihood cost. The default Gauss-Hermite quadrature inside `variational_expectation` does ~20–30 evaluations per datapoint per ELBO call. On Poisson count data, this dominates training time.

These are structurally independent — different modules, different math, different test surfaces — so we plan them as two PRs.

---

# PR 1 — Variational Fourier Features (1D, Matérn)

Hensman, Durrande, Solin 2017. A new `InducingVariables` subclass where inducing "features" are Fourier-basis projections on an interval. For Matérn kernels the prior covariance `Kuu` is diagonal plus low-rank, giving O(M) linear solves via Woodbury.

## Scope

**In.** 1D inputs. Matérn ½, 3/2, 5/2. SVGP and VFE wiring.

**Out, this PR.** Tensor-product 2D/3D extension. ExpQuad (different math). Spherical harmonic features. Kronecker interactions.

Rationale: the math is already per-kernel-nontrivial. Shipping 1D Matérn well is the killer feature; the generalisation is a follow-up.

## Architecture

Two extension points on existing abstractions, no changes to SVGP/VFE model bodies.

**(a) `InducingVariables` gains `K_uu(kernel)` and `K_uf(kernel, X)`.**
`Points(Z)` keeps current semantics via a default implementation calling `kernel(Z, Z)` and `kernel(Z, X)`. `FourierFeatures1D(a, b, num_frequencies)` overrides with VFF-specific closed forms.

**(b) `InducingVariables` gains an optional `solve_Kuu(kernel, rhs)`.**
Structured-matrix shortcut. `Points` doesn't override — callers fall back to `pt.linalg.solve(Kuu, rhs)`. `FourierFeatures1D` overrides with Woodbury on `Kuu = D + UUᵀ`, giving O(M) solves.

**Conditional dispatch.** `ptgp/conditionals.py` grows one branch: if `inducing` exposes `solve_Kuu`, use it; otherwise the current dense path. `base_conditional` signature unchanged.

## The PyTensor-rewrite question

CLAUDE.md says "write `pt.linalg.inv(K)` directly, let rewrites choose algorithms." But PyTensor's rewrite system doesn't currently exploit diagonal-plus-low-rank structure, and that structure is the entire performance story for VFF.

**Decision: thin bridge via `FourierFeatures1D.solve_Kuu`, using Woodbury explicitly.** Acceptable per CLAUDE.md's thin-bridge clause — the inducing class has private knowledge of its own Kuu structure; `solve_Kuu` is a kernel-agnostic interface, not a PyTensor-compiler workaround. File the general D+LR rewrite upstream as a follow-up issue.

Rejected alternatives: (i) dense Kuu punts on speed and the headline claim is absent at ship time; (iii) custom rewrite violates the "don't patch PyTensor" rule.

## API

```python
import ptgp as pg
import pytensor.tensor as pt

kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
likelihood = pg.likelihoods.Gaussian(sigma=sigma)
inducing = pg.inducing.FourierFeatures1D(a=0.0, b=10.0, num_frequencies=64)

gp = pg.gp.SVGP(kernel=kernel, likelihood=likelihood, inducing=inducing)
# compile_training_step, loop, compile_predict unchanged
```

`num_inducing = 2 * num_frequencies + 1` (DC + sines + cosines). `a, b` is the integration domain — must bracket the data. Add helper `FourierFeatures1D.from_data(X, num_frequencies, buffer=0.1)` for the common case.

## Tests

- **Correctness (primary).** VFF SVGP ELBO and predictions converge to exact GP as `num_frequencies → large` and domain covers data, on N=500 Matérn-½/3/2/5/2 draws. `atol=1e-3` — a convergence test, not an equivalence.
- **Correctness (secondary).** If GPflow VFF is present in a sibling checkout, cross-check `Kuu`, `Kuf`, and posterior mean at matched hyperparameters, `atol=1e-5`.
- **Woodbury correctness.** `FourierFeatures1D.solve_Kuu(rhs)` matches `pt.linalg.solve(Kuu_dense, rhs)` at `atol=1e-6`.
- **Speed sanity.** Single notebook benchmark: SVGP step time on N=50k, M=200 VFF vs M=200 Points. Documented, not asserted.

## Risks

- **Rewrite system materialising dense Kuu.** `FourierFeatures1D.K_uu` returns a structured container (`(diag, U)`), and the conditional dispatches on type. Dense Kuu is never formed unless explicitly requested.
- **Numerical conditioning at high M.** Fourier basis near edges blows up. Standard fix: reflective padding on `(a, b)`; document the `buffer` knob.
- **Recursion-limit gotcha.** Same `sys.setrecursionlimit(50000)` issue as current SVGP; no new contribution expected but verify during implementation.

---

# PR 2 — Analytic Variational Expectations (Poisson, exp link)

Replace Gauss-Hermite quadrature in `variational_expectation(y, μ, σ²)` with the exact closed form for Poisson with log link.

## Scope

Audit of what's actually closed form under Gaussian `q(f)`:

| Likelihood | Link | `E_q[log p(y|f)]` closed form? |
|---|---|---|
| Gaussian | identity | Already closed form in PTGP |
| **Poisson** | **exp** | **Yes, exact: `y·μ − exp(μ + σ²/2) − log(y!)`** |
| Poisson | softplus | No |
| Bernoulli | probit | No — tight approximations exist (Owen's T, Williams-Barber) but not 2-line |
| Bernoulli | logit | No (Jaakkola is a bound, different flavour) |
| StudentT | identity | No |
| NegBin | exp | No |

**This PR ships exactly one new analytic path: Poisson with log (exp) link.** Everything else stays on quadrature. Probit-Bernoulli is a deliberate exclusion — looks small from a distance, the honest version needs Owen's T, better as its own PR.

Rationale: count regression is a major applied GP workload (spatial epidemiology, ecology, traffic, call-centre modelling). Current quadrature cost dominates SVGP training at N≥10⁵. ~30 lines of code for a 20–30× per-step speedup on Poisson workloads.

## Architecture

**(a)** `Likelihood.variational_expectation(y, mu, var)` is the canonical entry point. Base class default uses Gauss-Hermite. `Poisson(link='exp')` overrides with the closed form.

**(b)** `Poisson` gets an explicit `link` kwarg (default `'exp'`, second option `'softplus'`). Only `'exp'` gets the analytic path; `'softplus'` stays on quadrature. Makes the link contract explicit.

No changes to SVGP, VFE, conditionals, inducing, or kernels.

## API

```python
likelihood = pg.likelihoods.Poisson(link="exp")      # analytic VE
likelihood = pg.likelihoods.Poisson(link="softplus") # quadrature VE
```

Existing user code that passes no `link` gets the fast path automatically (exp is the default). Compatible change.

## Tests

- **Analytic matches quadrature.** `variational_expectation` at `(y, μ, σ²)` grid points, analytic vs 64-point Gauss-Hermite, `atol=1e-6`.
- **Gradient correctness.** JAX autograd of the analytic expression matches finite differences w.r.t. μ and σ².
- **End-to-end.** SVGP on Poisson count data with `link='exp'` converges to the same ELBO and posterior mean as quadrature, `atol=1e-4`.
- **Speed sanity.** Notebook benchmark: single-step ELBO time on N=50k Poisson counts, analytic vs quadrature.

## Risks

- **`log(y!)` overflow for large counts.** Use `pt.gammaln(y + 1)`.
- **Silent divergence from quadrature path.** Guaranteed by the test battery above; cross-check against GPflow Poisson if a sibling checkout is present.

---

## PR ordering

Independent — disjoint files (`ptgp/likelihoods/poisson.py` vs `ptgp/inducing.py` + `ptgp/conditionals.py` + kernel-method additions). Either order works. Suggested: **PR 2 first** as the review-warmup, then PR 1.
