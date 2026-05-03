# Rewrite analysis — `Unapproximated` GP, joint forward + gradient

How the PyTensor rewrite system lowers `gp.Unapproximated` + `marginal_log_likelihood` plus its gradient when both are compiled in a single `pytensor.function` call (which is what PTGP's training entry points actually do). Identifies the gap between today's compiled graph and the theoretical floor, and locates the two structural blockers in PyTensor's optimisation pipeline that are still keeping us above the floor.

VFE and SVGP have separate dynamics and are not covered here.

PyTensor commit: `jessegrabowski/pytensor@assumption-system` (`a7394bd`).

---

## 1. What we measure and why

The unit of analysis is

```python
pytensor.function([X, y, σ, ℓ], [loss, ∂loss/∂σ, ∂loss/∂ℓ])
```

because that is what PTGP actually compiles at training time:

- `ptgp/optim/training.py::compile_scipy_objective` calls `pt.grad(loss, theta)` and passes `[loss, flat_grad]` as outputs to one `pytensor.function`.
- `compile_training_step` does the same indirectly via `updates={param: param - lr · grad(loss, param)}` — every gradient is forced into the same `FunctionGraph`.

Joint compilation is what lets shared subexpressions (most importantly `Cholesky(K_noisy)`) be a single `Apply` node feeding both forward and backward consumers. Forward-only counts hide whether sharing actually happens; gradient-only counts double-count things the forward already computes. The joint number is the only one that maps to wall-clock cost per training step.

We look at exactly two graph states:

- **Pre-rewrite**: the user-written symbolic graph. Counted by walking `ancestors([loss, *grads])` over `var.owner.op` types.
- **Post-`fast_run`**: the compiled `FunctionGraph`. Counted from `fn.maker.fgraph.apply_nodes`. This is the union of `canonicalize`, `stabilize`, `specialize`, plus `FusionOptimizer` and `InplaceElemwiseOptimizer`.

Per-stage breakdowns (`canonicalize` / `stabilize` / `specialize` only) show up only when needed to localise a specific rewrite — they're a diagnostic, not the headline.

---

## 2. The model

Exact GP with a Gaussian likelihood:

$$
y \mid X, \theta \sim \mathcal{N}\!\bigl(\mu(X),\; K_{ff}(X, X) + \sigma^2 I\bigr).
$$

Loss is the negative log marginal likelihood. Writing $K \equiv K_{ff} + \sigma^2 I$ and $b \equiv y - \mu(X)$:

$$
L(\theta) = -\tfrac{1}{2}\Bigl(\,b^\top K^{-1} b \;+\; \log\lvert\det K\rvert \;+\; N \log 2\pi\Bigr).
$$

Gradient (closed form via L_op of `MatrixInverse` and `SLogDet`):

$$
\frac{\partial L}{\partial \theta_i}
\;=\; -\tfrac{1}{2}\,\mathrm{tr}\!\left(K^{-1}\frac{\partial K}{\partial \theta_i}\right)
\;+\; \tfrac{1}{2}\, b^\top K^{-1}\frac{\partial K}{\partial \theta_i} K^{-1} b.
$$

So the gradient needs **the same** `K^{-1} b` the forward needed, plus access to `K^{-1}` for the trace term. Both can be served by one Cholesky factor `L = chol(K)`.

What the user actually wrote (`ptgp/objectives.py::marginal_log_likelihood`):

```python
K = gp.kernel(X) + gp.likelihood.sigma**2 * pt.eye(X.shape[0])
sign, logdet = pt.linalg.slogdet(K)
K_inv = pt.linalg.inv(K)
return -0.5 * (diff @ K_inv @ diff + logdet + N * pt.log(2.0 * pt.pi))
```

Two design choices in PTGP relevant to the analysis:

- The kernel output is wrapped in `pt.assume(K, symmetric=True, positive_definite=True)` inside `Kernel.__call__`.
- `Gaussian.sigma` is wrapped in `pt.assume(sigma, positive=True)` inside `Gaussian.__init__` (this is one of the rules we added — upstream's `pt.assume` doesn't accept `positive=`; ptgp patches it locally to do so).

These assumptions are what make the Solve / SLogDet / etc. lowerings reachable.

---

## 3. The ideal compiled joint graph

If a human optimised the joint graph, it would look like:

```
L = Cholesky(K_noisy)                    ← one O(N³) factorisation
α = CholeskySolve(L, b)                  ← one O(N²) solve

forward
  b·K⁻¹·b   = b · α
  log|detK| = 2 · sum(log(diag(L)))

gradient (per parameter θ_i)
  K⁻¹·b·b·K⁻¹  uses the same α
  tr(K⁻¹·dK/dθ_i)  uses CholeskySolve(L, dK/dθ_i) (or just diag(K⁻¹) when dK/dθ_i is diagonal)
```

**Cubic-factorisation floor: 1 per training step.** Everything else is `O(N²)` triangular solves driven by a single shared `L`.

---

## 4. Joint op counts: pre-rewrite vs post-`fast_run`

Counting ops on `[loss, ∂loss/∂σ, ∂loss/∂ℓ]`. ✓ marks numbers that match the floor; **bold** marks the gap.

|  | Cholesky | CholeskySolve | SolveTri | Solve | SLogDet | Det | LUFactor | MatrixInverse |
|---|---|---|---|---|---|---|---|---|
| pre-rewrite (raw symbolic) | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 3 |
| post-`fast_run`, ptgp rules **on** | **1** ✓ | 7 | 0 | 0 | 0 | 0 | 0 | 0 ✓ |
| post-`fast_run`, ptgp rules **off** (baseline) | 0 | 0 | 10 | 0 | 0 | 0 | **2** | **2** |
| ideal | **1** ✓ | ≥ 1 | 0 | 0 | 0 | 0 | 0 | 0 ✓ |

Cubic-factorisation count per training step (Cholesky + LUFactor + MatrixInverse + Solve, all `O(N³)` internally):

- pre-rewrite: 5 (3 inverses + 2 dets, the dets being lowered to LU later)
- post-`fast_run` with ptgp: **1** ✓ (one Cholesky, reused everywhere — at the floor)
- post-`fast_run` without ptgp: **4** (2 LUFactors + 2 MatrixInverses)
- ideal: **1**

So with ptgp's rules in place the joint graph does **a quarter** of the cubic work of the baseline and reaches the theoretical floor of one Cholesky factorisation per training step.

### What the existing ptgp rules already give us

- **Forward-only gain.** On the forward path alone the rules produce a clean `Cholesky=1, CholeskySolve=1, MatrixInverse=0` lowering (see `ptgp/rewrites.py` and `tests/test_rewrites.py`).
- **Joint cubic-op reduction (4 → 1).** Five rewrites cooperate:
  - `psd_solve_to_chol_solve` (upstream PyTensor) lowers `Solve(K_noisy, b) → CholeskySolve(L, b)`. Fires *only* if `K_noisy` is recognised as PSD, which depends on ptgp's `POSITIVE` assumption rules.
  - `slogdet_specialize` (in `ptgp/rewrites.py`) lowers `SLogDet(K_noisy)` and reuses the Cholesky from above via `_existing_cholesky`.
  - `matrix_inverse_specialize` (in `ptgp/rewrites.py`) lowers the standalone `MatrixInverse(K_noisy)` Apply nodes that `pt.grad(slogdet)` emits, also reusing the existing Cholesky. Drops the joint count from 4 cubics to 2.
  - `merge_composites_with_shared_inputs` (in `ptgp/rewrites.py`) deduplicates the kernel `Elemwise(Composite)` Apply that `FusionOptimizer` would otherwise create twice (once per convex closure on the forward and gradient paths). Drops the joint count from 2 cubics to 1.
  - `merge_after_composite_dedup` (an extra `MergeOptimizer` pass in `ptgp/rewrites.py`) collapses the now-equivalent downstream `SpecAssum`/`Add`/`Cholesky`/`MatrixInverse` Apply nodes that the Composite-merge made structurally identical. PyTensor's stock pipeline only runs `MergeOptimizer` at position 49 (before fusion), so without this second pass the duplicates persist into inplace optimization and get permanently locked in by differing `inplace_pattern`s.
- **`POSITIVE` assumption + propagation rules** (`Sqr`/`Pow`/`Mul`/`AllocDiag`/`set_subtensor`/transpose-of-PSD/quadratic-form `Dot`) are what make `K_noisy = K + σ²I` get recognised as PSD in the first place. Without them, `Solve(K_noisy, b)` goes through generic LU paths — the without-ptgp row shows `LUFactor=2, SolveTriangular=10` instead of `Cholesky=1, CholeskySolve=7`.
- **No `Det` survives in either case.** Pre-rewrite has 2 `Det` Applies; both with-ptgp and without-ptgp show `Det=0` post-`fast_run`. With ptgp, our `slogdet_specialize` does the elimination by routing through the Cholesky. Without ptgp, an upstream rewrite (`det_of_matrix_factorized_elsewhere` in `pytensor/tensor/rewriting/linalg/summary.py`) does the elimination by routing through the LUFactor that was created by the Solve lowering. Either way the slogdet path adds no extra cubic cost — it piggybacks on the factorisation that already exists.

---

## 5. Blocker A — duplicated kernel `Composite` from greedy fusion  *(fixed)*

**Status:** Fixed locally. Implementation: `merge_composites_with_shared_inputs` + an extra `MergeOptimizer` pass (`merge_after_composite_dedup`) in `ptgp/rewrites.py`, registered into `optdb` at top-level positions 49.7 and 49.75. Tests: `tests/test_rewrites.py::test_merge_composites_*`.

### Cause

Pre-fusion, the joint graph has **one** `Cholesky(K_noisy)` and **one** `MatrixInverse(K_noisy)`. Verified by compiling with fusion excluded:

```
mode.excluding('fusion')  →  Cholesky=1, MatrixInverse=1, CholeskySolve=3
```

`FusionOptimizer` (in `pytensor/tensor/rewriting/elemwise.py`) then greedily forms maximal *convex* Elemwise subgraphs and rewrites each into a single `Elemwise(Composite)` `Apply`. The forward and gradient paths happen to have *different* convex closures around the kernel:

- Forward subgraph: only the kernel value `exp(-½·max(d², 0))` is needed externally → fusion produces a 1-output `Composite{exp(...)}`.
- Gradient subgraph: the chain rule needs intermediate `Maximum(...)` and `add(...)` values too → fusion produces a 3-output `Composite` exposing all three.

So fusion creates **two `Elemwise(Composite)` `Apply` nodes that share inputs (id-equal) but have different output signatures**. `MergeOptimizer` keys on `(op, inputs)` and the two Composite ops are not `__eq__` (different inner fgraph outputs → different `__props__`), so they cannot merge.

The duplication then propagates upward: each Composite output gets its own `SpecifyAssumptions{PSD,sym}` wrapper (different `Variable`s), each wrapper feeds its own `Add(SpecAssum, σ²·diagonal)`, and each `Add` feeds its own `Cholesky` *and* its own `MatrixInverse`.

### The inplace problem on top of fusion

Once the two `Composite` Applies exist, `InplaceElemwiseOptimizer` (registered at top-level `optdb` position **49.5**, immediately after `FusionOptimizer` at **49**) runs and assigns the duplicated downstream `Add` Applies *different* inplace destinations:

```
Add[0].op = Elemwise(scalar_op=add, inplace_pattern=<frozendict {0: 0}>)
Add[1].op = Elemwise(scalar_op=add, inplace_pattern=<frozendict {0: 1}>)
op_equal: False     ← because inplace_pattern differs
inputs_equal: True
```

After that, even a perfect post-hoc merge of the Composites can't unify the downstream chain — the ops are no longer `__eq__`.

Empirical confirmation: with a manual Composite merge (build a new multi-output Composite with inner CSE + slot dedup), and then run outer `MergeOptimizer`:

| mode | Cholesky | MatrixInverse |
|---|---|---|
| default mode, manual merge | 2 | 2 |
| `mode.excluding('inplace')`, manual merge | **1** | **1** |

So the approach works; the fix just has to land **before** inplace. Numerical correctness of the manual merge was verified at `max diff ≈ 7e-15` against an unmodified reference compile (well within the snippet's `atol=1e-12`; the residual is plain float64 roundoff).

### Pre-fix cost

+1 cubic factorisation per training step from the duplicate `Cholesky`, plus +1 per step from the duplicate `MatrixInverse` (Blocker B). With Blocker B already fixed but Blocker A still open, the joint graph had `Cholesky=2, MatrixInverse=0`.

### What we did

Two cooperating passes, registered together because they have to bracket the gap PyTensor's stock pipeline leaves between `FusionOptimizer` and the inplace optimizers:

1. **`merge_composites_with_shared_inputs`** (a `node_rewriter([Elemwise])` filtered to `Elemwise(Composite)`). For each Composite Apply `A`, walk `fgraph.clients[A.inputs[0]]` for a sibling `Elemwise(Composite)` `B` with id-identical inputs. Build a merged inner scalar fgraph by remapping `B.scalar_op.inputs` onto `A.scalar_op.inputs` (`clone_replace`), then run `MergeOptimizer` on that inner fgraph for inner CSE. Deduplicate output **slots** that collapse to the same inner Variable. Construct one `Composite(scalar_inputs, unique_outputs)`, materialise with `Elemwise(scalar_op=merged)(*inputs, return_list=True)`, and return a `{old_var: new_var}` dict mapping both `A.outputs` and `B.outputs` onto the merged outputs.

2. **`merge_after_composite_dedup`**, an extra `MergeOptimizer()` pass. The Composite-merge above leaves the downstream `SpecAssum`/`Add`/`Cholesky`/`MatrixInverse` Apply nodes structurally identical (same op, same inputs by id) but they're still distinct Apply objects. Without a follow-up `MergeOptimizer`, they survive into `InplaceElemwiseOptimizer` and get permanently locked in by differing `inplace_pattern`s.

Registered at top-level `optdb` positions **49.7** and **49.75**, after `add_destroy_handler` (49.5) and before any inplace pass (earliest is `blockwise_inplace` at 50.10).

### Effect on the joint graph

| state | Cholesky | CholeskySolve | MatrixInverse | cubic ops |
|---|---|---|---|---|
| pre-fix (with Blocker B already in place) | 2 | 7 | 0 | 2 |
| post-fix (both blockers fixed) | **1** ✓ | 7 | 0 | **1** ✓ |

The single `Cholesky(K_noisy)` now feeds the forward `CholeskySolve(L, b)`, the forward `Prod(diag(L))` for `log|det K|`, and the gradient's two `CholeskySolve(L, eye)` calls (Blocker B's lowering of the standalone `MatrixInverse` Apply nodes). One factorisation, reused everywhere.

Numerical correctness: loss diff vs analytic reference `1.4e-14`; gradient diff vs finite-difference `~1.5e-9`. Verified end-to-end and in `tests/test_rewrites.py::test_merge_composites_*`.

### Where this lives

- **Local landing spot:** `ptgp/rewrites.py`. The rewrite is installed into `optdb` between top-level positions 49 (`FusionOptimizer`) and 50.10 (the earliest inplace pass), via `optdb.register("name", in2out(rewriter, ignore_newtrees=True), "fast_run", position=49.7)`. The follow-up `MergeOptimizer` is registered the same way at position 49.75.

- **Upstream landing spot:** inside `fuse_seqopt` in `pytensor/tensor/rewriting/elemwise.py`, registered between `FusionOptimizer` (sub-position 1) and `local_useless_composite_outputs` (sub-position 2). The follow-up `MergeOptimizer` could either live in the same `fuse_seqopt` (as sub-position 3, after `local_useless_composite_outputs`) or be promoted to a top-level `optdb` entry next to the existing `merge2` (which currently runs only at position 49, before fusion). The Composite-merge is a generic `FusionOptimizer` companion and has nothing to do with GPs — it should make sense to anyone who hits the same convex-closure duplication pattern.

---

## 6. Blocker B — `MatrixInverse(PSD A)` from `pt.grad(slogdet)`  *(fixed)*

**Status:** Fixed locally. Implementation: `matrix_inverse_specialize` in `ptgp/rewrites.py`. Tests: `tests/test_rewrites.py::test_matrix_inverse_*`.

### Cause

`pt.grad`'s L_op for `SLogDet(A)` emits `inv(A).T` as the cotangent (mathematically correct — `d/dA log|det A| = (A^{-T})`). In the joint graph this lands as a standalone `MatrixInverse(K)` `Apply` whose output is then consumed by something else in the gradient computation.

PyTensor has `psd_solve_to_chol_solve` (in `pytensor/tensor/rewriting/linalg/solvers.py`) that lowers `Solve(PSD A, b) → CholeskySolve(L, b)`. Before this PR there was **no analogous rewrite for `MatrixInverse`**, because materialising a dense inverse is rare in user code and almost always undesirable — except here, where it's emitted by autodiff and immediately contracted with another tensor.

Inspecting the joint graph's `MatrixInverse` Apply nodes pre-fix:

```
MatrixInverse(K=Elemwise)  consumer ops: ['Elemwise']      ← contracted via a Composite for the b·K⁻¹·dK/dθ·K⁻¹·b term
MatrixInverse(K=Elemwise)  consumer ops: ['ExtractDiag']   ← diagonal extracted for tr(K⁻¹·dK/dθ) when dK/dθ is diagonal (the σ-grad)
```

Both consumers contract against an N×N tensor (the Composite contracts a fused expression involving K⁻¹; ExtractDiag pulls out N diagonal entries). Neither narrows the right-hand side, so a consumer-specific rewrite would not save cubic cost beyond what a uniform `MatrixInverse → cho_solve(L, eye)` rewrite already gives.

### Pre-fix cost

+1 cubic factorisation per `K_noisy` Apply — currently +2 per training step (because Blocker A produces two `K_noisy` instances).

### What we did (Option A from the design note)

A `node_rewriter` keyed on `Blockwise(MatrixInverse)`. For each `MatrixInverse(A)` Apply where `check_assumption(fgraph, A, POSITIVE_DEFINITE)`:

1. Walk `fgraph.clients[A]` for an existing `Cholesky(A)` Apply via the same `_existing_cholesky` helper used by `slogdet_specialize`. Reuses the factor if present.
2. If no Cholesky exists yet, materialise `cholesky(A, lower=True)` — same cubic cost as the original `MatrixInverse`, but the Cholesky is now reusable by other consumers (and by Blocker A's merge once that lands).
3. Emit `cho_solve((L, True), eye(N, dtype=A.dtype))` and return as the replacement for the MatrixInverse output.

The choice not to pattern-match the consumer (`Dot(inv, X) → cho_solve(L, X)`, etc.) is deliberate: in this graph both consumers want the full inverse anyway, and uniform replacement keeps the rewrite to ~10 lines.

### Effect on the joint graph

| state | Cholesky | CholeskySolve | MatrixInverse | cubic ops |
|---|---|---|---|---|
| pre-fix | 2 | 5 | 2 | 4 |
| post-fix | 2 | 7 | 0 | 2 |

The two `MatrixInverse` Apply nodes are gone; the gained `CholeskySolve` Apply nodes (against `eye`) reuse the existing Cholesky factors, so no new cubic factorisations were introduced. Verified numerically in `test_matrix_inverse_lowering_is_numerically_correct` and end-to-end against an unmodified reference compile of the GP MLL joint graph (loss diff `1.4e-14`; finite-difference gradient diff `1.5e-9`).

### Where this lives

- **Local landing spot:** `ptgp/rewrites.py`, registered at `register_specialize` next to `slogdet_specialize`. Reuses `_existing_cholesky` and the `POSITIVE_DEFINITE` assumption framework.
- **Upstream landing spot:** `pytensor/tensor/rewriting/linalg/inverse.py`, alongside the existing `MatrixInverse` rewrites. There is precedent (the file already specialises `MatrixInverse` for orthogonal inputs in `a7394bd`); a PSD specialisation is the natural sibling, and the rewrite as written depends only on PyTensor APIs.

---

## 7. Summary

| state | cubic factorisations / step |
|---|---|
| pre-rewrite | 5 (Det×2 + Inv×3) |
| ptgp rules **off** (baseline) | **4** (LU×2 + Inv×2) |
| ptgp rules **on**, today (both blockers fixed) | **1** ✓ |

The joint graph for `Unapproximated` is at the floor: one Cholesky factorisation per training step, reused by every linear-algebra consumer in both the forward and gradient subgraphs. Reaching this took five cooperating rewrites: ptgp's `POSITIVE` propagation rules feed PyTensor's `psd_solve_to_chol_solve`, plus four ptgp-local rewrites — `slogdet_specialize`, `matrix_inverse_specialize`, `merge_composites_with_shared_inputs`, and the follow-up `merge_after_composite_dedup`.

---

## 8. VFE follow-ups

VFE's joint graph is now at the **2** cubic-op floor (`Cholesky(Kuu)` and `Cholesky(D)` where `D = σ²·Kuu + Kuf @ Kuf.T`). Reaching it took two steps: closing Blocker C (a missing PSD-quadratic-form recognition) and reformulating `collapsed_elbo` itself with the Woodbury identity (so the N×N covariance is never materialised). Blocker D — `Dot`-level CSE for the duplicated Q matrix — is now *obsolete*: the Woodbury reformulation routes around it by never producing the pattern.

### Blocker C — PSD propagation through `Dot(X.T, CholeskySolve(L, X))`  *(fixed)*

**Status:** Fixed locally. Implementation: extra clause in `_dot_quadratic_form_psd` in `ptgp/rewrites.py` (commit `f9e7b71`). Tests: `tests/test_rewrites.py::test_quadratic_form_with_cholesky_solve_is_psd`.

#### Cause

ptgp's `_dot_quadratic_form_psd` rule originally recognized `Dot(X.T, Solve(M, X))` as PSD (the canonicalised form `X.T · M⁻¹ · X`). On VFE's gradient path, however, Blocker B's `matrix_inverse_specialize` rewrite has already lowered the inner `MatrixInverse` to a `CholeskySolve`, producing `Dot(X.T, CholeskySolve(L, X))` instead. The original rule missed that form. Result: the gradient-side `Q + σ²I` matrix wasn't PSD-recognized, so `psd_solve_to_chol_solve` and `matrix_inverse_specialize` didn't fire on it — the standalone `MatrixInverse` Apply nodes survived.

#### Pre-fix cost

+1 cubic factorisation per gradient-side `Q + σ²I` Apply. The joint graph has two such Apply nodes (Blocker D — see below), so the unfixed cost was +2 cubic ops/step.

#### What we did

Added a separate rule `_dot_xt_chosolve_x_psd` (after the original `_dot_quadratic_form_psd` was split into three focused rules during a later refactor): if `b = CholeskySolve(L, X2)` and `X2 is X` and `L` is the output of a `Cholesky` Apply (so factored a PSD matrix by construction), conclude `X.T @ CholeskySolve(L, X) ≡ X.T @ M⁻¹ @ X` is PSD. Uses generic `_matches_core_op` and `_core_op_of` helpers in `ptgp/rewrites.py`.

#### Effect on the joint graph

| state | Cholesky | CholeskySolve | MatrixInverse | cubic ops |
|---|---|---|---|---|
| pre-fix | 3 | 10 | 2 | 5 |
| post-fix | 3 | 12 | 0 | **3** |

Both gradient-side `MatrixInverse` Apply nodes lower (the analysis above predicted "+1 cubic op savings", but two were observed empirically because Blocker D's duplication produces *two* gradient-side `Q + σ²I` instances, both of which now lower). Numerical correctness verified end-to-end: the with-Blocker-C and without-Blocker-C joint graphs agree on `loss`, `g_sigma`, `g_ls`, `g_Z` to within float64 noise (≤ 4.3e-6, identical spread to the no-Blocker-C-vs-baseline residual — Blocker C did not introduce instability).

#### Where this lives

- **Local landing spot:** `ptgp/rewrites.py`, as `_dot_xt_chosolve_x_psd` (one of three sibling rules after splitting the original combined `_dot_quadratic_form_psd`).
- **Upstream landing spot:** `pytensor/tensor/rewriting/linalg/solvers.py` (next to the `Solve`-form rule), or wherever the upstream PSD-quadratic-form recognizer eventually lives. The rule is generic — any chain `X.T @ M⁻¹ @ X` lowered through Cholesky benefits.

### Blocker D — `Dot`-level duplication of the Q matrix  *(obsolete)*

**Status:** Obsolete. The original blocker: with the naive `collapsed_elbo`, the Q matrix `Kuf.T @ Kuu⁻¹ @ Kuf` was materialised twice in the joint graph (different `Dot` Apply ids) — once on the forward path, once on the gradient path. The fix would have been a `Dot`-level CSE pass run after `psd_solve_to_chol_solve` (when the two chains finally become structurally equivalent).

That fix is no longer needed because `collapsed_elbo` was reformulated via the Woodbury identity (commit `99f9803`). The new formulation operates on the M×M matrix `D = σ²·Kuu + Kuf @ Kuf.T` instead of the N×N matrix `cov = σ²I + Q`, so Q itself is never materialised. The duplication-via-gradient-pullback pattern no longer arises, and VFE drops to the 2-cubic-op floor without needing a `Dot`-level CSE pass.

The `Dot`-level CSE rewrite would still be a useful generic addition — any future objective with a similar shared-Dot pattern would benefit. But it's no longer on the critical path for VFE. Recorded in `FUTURE.md` as a candidate enhancement when a second pattern surfaces.

### VFE post-Woodbury joint graph

| | Cholesky | CholeskySolve | MatrixInverse | cubic ops |
|---|---|---|---|---|
| pre-rewrite (Woodbury form) | 0 | 0 | 6 | 10 (Det×4 + Inv×6) |
| post-`fast_run`, ptgp on | **2** ✓ | 10 | 0 | **2** ✓ |
| post-`fast_run`, ptgp off (baseline) | 2 | 10 | 0 | 2 |

Numerical: `ptgp_on` and `ptgp_off` agree bit-identically (0.0e+00 across loss and all gradients) — the deterministic compile-graph structure means rewrite ordering is irrelevant. Loss agrees with a stable numpy reference to ~1e-9 relative error; this stays roughly constant as N grows (vs the old naive form whose error grew with N because the N×N `cov` became increasingly ill-conditioned).

---

## 9. SVGP follow-ups

`SVGP` + `elbo` previously sat at **2** cubic factorizations/step (one above the floor of 1). The wasteful Cholesky was on `q_sqrt @ q_sqrt.T` from `gauss_kl`'s `slogdet(q_cov)` lowering. With Blocker E closed (see below), SVGP's joint graph now hits the floor.

| | pre-Blocker-E | post-Blocker-E (current) | ideal |
|---|---|---|---|
| Cholesky | 2 | **1** ✓ | 1 |
| Det | 1 | **0** ✓ | 0 |
| MatrixInverse | 0 | 0 | 0 |
| **cubic ops** | 3 | **1** ✓ | **1** |

### Blocker E — `Cholesky(L @ L.T)` and friends when `L` is the variational Cholesky factor  *(fixed)*

**Status:** Fixed locally via three cooperating changes:
1. SVGP's variational parameters are now constructed via `init_variational_params(M)` returning a `VariationalParams` dataclass. ``q_sqrt`` is stored as a flat vector of length `M·(M+1)/2` and materialized as `_softplus_lower_triangular(flat, M)` — guaranteeing lower-triangular with strictly positive diagonal at every optimizer step. The `lower_triangular=True` assumption is now *sound* (previously it was a lie under the bare-matrix parameterization, since the optimizer's gradient updates filled the upper triangle).
2. Three new ptgp rewrites that fold operations on `L @ L.T` to `L`-direct expressions:
   - `slogdet_specialize` (extended with an `L @ L.T` fast path): `SLogDet(L @ L.T) → [1, 2·sum(log|diag(L)|)]`. Eliminates the wasteful Cholesky on the forward path.
   - `matrix_inverse_specialize` (extended with an `L @ L.T` fast path): `MatrixInverse(L @ L.T) → cho_solve((L, True), eye)` — uses `L` directly as the Cholesky factor instead of computing a fresh `Cholesky(L @ L.T)`. Eliminates the wasteful Cholesky on the gradient path.
   - `det_of_LLT_to_diag_product` (new): `Det(L @ L.T) → (prod(diag(L)))**2`. Eliminates the standalone `Det` Apply that survives `slogdet_specialization` when `Det` is also referenced by the gradient pullback.
3. New rewrite `diag_of_AAT_to_row_norms_squared`: `ExtractDiag(A @ A.T) → sum(A**2, axis=-1)`. Generic identity (works for any `A`), no assumption needed. Eliminates the `M^2`-element materialization of `q_cov = q_sqrt @ q_sqrt.T` that `pt.trace(q_cov)` would otherwise produce.

All in `ptgp/rewrites.py`; tests in `tests/test_rewrites.py::test_slogdet_of_LLT_*`, `test_det_of_LLT_*`, `test_diag_of_AAT_*`.

#### Cause

`gauss_kl(q_mu, q_sqrt)` materializes `q_cov = q_sqrt @ q_sqrt.T` and calls `slogdet(q_cov)` and `pt.trace(q_cov)` on it. PyTensor's `slogdet_specialize` then lowered the slogdet to a Cholesky on `q_cov` — wasteful, since `q_sqrt` *is* the Cholesky factor by construction. Similarly the `pt.grad(slogdet)` pullback emitted a `MatrixInverse(q_cov)` which was lowered through Cholesky on `q_cov` again. And `pt.trace(q_cov)` materialized the full `M×M` outer product just to extract its diagonal.

The pre-fix design also had a soundness bug: `q_sqrt` was just a free `pt.matrix` parameter annotated `lower_triangular=True`, but gradient updates fill the upper triangle, so the assumption was a lie within a few training steps. The fix's flat-vector + softplus parameterization makes the assumption correct by construction.

#### Pre-fix cost

+1 cubic factorisation per training step from the wasteful `Cholesky(q_sqrt @ q_sqrt.T)`, plus +1 cubic op from the standalone `Det(q_sqrt @ q_sqrt.T)` (left over from `slogdet_specialization` failing to fire when both `Sign(Det)` and the gradient's `Mul(Det, ...)` consume it), plus +1 `M^2` matmul from the trace term materialization.

#### What we did (collapse)

Together the changes: (a) make the `lower_triangular=True` annotation sound; (b) augment `slogdet_specialize` and `matrix_inverse_specialize` with an `L @ L.T` fast path that uses `L` directly; (c) add `det_of_LLT_to_diag_product` and `diag_of_AAT_to_row_norms_squared` rewrites for the Det and trace cases respectively.

#### Effect on the joint graph

| state | Cholesky | CholeskySolve | Det | MatrixInverse | cubic ops |
|---|---|---|---|---|---|
| pre-Blocker-E | 2 | 1 | 1 | 0 | 3 |
| post-Blocker-E | **1** ✓ | 1 | **0** ✓ | 0 | **1** ✓ |

One Cholesky factorisation per training step (`Cholesky(Kuu)`), reused everywhere. `gauss_kl` is unchanged — all the optimization happens in the rewrite layer.

Numerical correctness verified end-to-end: SVGP+Bernoulli classification accuracy remains > 0.85 over 400 Adam steps, and the cross-library ELBO match against GPJax holds at `atol=1e-5`. Full test suite (154 tests) passes.

#### Where this lives

- **Local landing spot:** `ptgp/rewrites.py`. The slogdet/MatrixInverse augmentations are inline modifications to existing `register_specialize` rewrites; the new `det_of_LLT_to_diag_product` and `diag_of_AAT_to_row_norms_squared` are sibling `register_specialize` rules.
- **Upstream landing spot:** `pytensor/tensor/rewriting/linalg/summary.py` (slogdet/det helpers) and `pytensor/tensor/rewriting/linalg/inverse.py` (matrix-inverse). The `diag_of_AAT_to_row_norms_squared` rewrite is fully generic — no GP context — and is a natural sibling for any "trace of outer product" identity in PyTensor's linear-algebra rewrite suite.

### Summary across all three GP types

| objective | floor | ptgp today | gap | blockers remaining |
|---|---|---|---|---|
| `Unapproximated` | 1 | **1** ✓ | none | — |
| `VFE`            | 2 | **2** ✓ | none | — |
| `SVGP`           | 1 | **1** ✓ | none | — |

All three forward+gradient training graphs hit their cubic-factorisation floor. Pinned by `tests/test_cubic_floor.py`; full per-op breakdown via `scripts/joint_graph_analysis.py`. No remaining blockers; the closest open item is the inplace miss on `Cholesky(SpecifyAssumptions(Kuu))` in SVGP (a memory issue, not a cubic-op issue) — see `FUTURE.md`.

---

## Appendix A — diagnostic toolkit

The code snippets used to produce every number above. Drop into a Python session with `ptgp` importable.

### A.1 Setup: build the joint graph

```python
import pytensor
import pytensor.tensor as pt
import ptgp
from ptgp.gp import Unapproximated
from ptgp.kernels import ExpQuad
from ptgp.mean import Zero
from ptgp.objectives import marginal_log_likelihood

X = pt.dmatrix('X'); y = pt.dvector('y')
ls = pt.dscalar('ls'); sigma = pt.dscalar('sigma')
gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
loss = marginal_log_likelihood(gp, X, y)
g_sigma, g_ls = pt.grad(loss, [sigma, ls])
joint = [loss, g_sigma, g_ls]
```

### A.2 Op-counting helpers

The two states need different counters because pre-rewrite is symbolic and post-`fast_run` is a `FunctionGraph`:

```python
import collections
from pytensor.graph.traversal import ancestors
from pytensor.tensor.blockwise import Blockwise

def op_name(op):
    # Unwrap Blockwise so e.g. Blockwise(Cholesky) counts as Cholesky.
    return type(op.core_op).__name__ if isinstance(op, Blockwise) else type(op).__name__

def count_symbolic(outs):
    """Pre-rewrite counts on the user's symbolic outputs."""
    c = collections.Counter()
    for v in ancestors(outs):
        if v.owner is not None:
            c[op_name(v.owner.op)] += 1
    return c

def count_compiled(fn):
    """Post-fast_run counts on the compiled FunctionGraph."""
    return collections.Counter(op_name(n.op) for n in fn.maker.fgraph.apply_nodes)
```

### A.3 Toggling `ptgp.rewrites` to get the baseline column

`ptgp.rewrites` registers its assumption rules into the global `ASSUMPTION_INFER_REGISTRY` at import time. To get a "what would PyTensor do without ptgp" baseline, snapshot the registry, strip out anything from `ptgp.rewrites`, run the comparison, restore:

```python
from pytensor.tensor.assumptions.core import ASSUMPTION_INFER_REGISTRY
import ptgp.rewrites as R

ptgp_funcs = {
    getattr(R, n) for n in dir(R)
    if callable(getattr(R, n))
    and getattr(getattr(R, n), '__module__', None) == 'ptgp.rewrites'
}
snap = {k: list(v) for k, v in ASSUMPTION_INFER_REGISTRY.items()}

def disable_ptgp():
    for k, fns in ASSUMPTION_INFER_REGISTRY.items():
        ASSUMPTION_INFER_REGISTRY[k] = [f for f in fns if f not in ptgp_funcs]

def restore_ptgp():
    for k in list(ASSUMPTION_INFER_REGISTRY.keys()):
        del ASSUMPTION_INFER_REGISTRY[k]
    for k, v in snap.items():
        ASSUMPTION_INFER_REGISTRY[k] = list(v)
```

ptgp's `slogdet_specialize` and `matrix_inverse_specialize` `node_rewriter`s are registered at module-import time too, but they gate on `check_assumption(fgraph, A, POSITIVE_DEFINITE)` — with the assumption rules stripped, no input resolves PSD outside the bare kernel-output annotation and these rewrites trivially never fire.

The two **post-fusion** passes (`merge_composites_with_shared_inputs` and `merge_after_composite_dedup`, registered into `optdb` at top-level positions 49.7 and 49.75) do *not* gate on assumption rules — they're pure structural rewrites and run regardless. To get a true "PyTensor without any of ptgp's contributions" baseline, exclude them via the compile mode:

```python
mode_no_ptgp_optdb = pytensor.compile.mode.get_default_mode().excluding(
    "merge_composites_with_shared_inputs",
    "merge_after_composite_dedup",
)
```

Pass that as `mode=` to `pytensor.function` *while* `disable_ptgp()` is in effect to reproduce the "ptgp off" row in §4.

### A.4 The headline three-row table

```python
mode_no_ptgp_optdb = pytensor.compile.mode.get_default_mode().excluding(
    "merge_composites_with_shared_inputs",
    "merge_after_composite_dedup",
)

print('pre-rewrite:                ', count_symbolic(joint))
fn = pytensor.function([sigma, ls, X, y], joint)
print('post-fast_run, ptgp on:     ', count_compiled(fn))
disable_ptgp()
fn = pytensor.function([sigma, ls, X, y], joint, mode=mode_no_ptgp_optdb)
print('post-fast_run, ptgp off:    ', count_compiled(fn))
restore_ptgp()
```

### A.5 Blocker-A diagnostics (now no-op post-fix)

These were the snippets that originally diagnosed Blocker A. Post-fix, the joint graph has only one `Cholesky`, so the sibling-Composite scan finds nothing on the live graph. The pre-fix outputs are preserved here for context — to re-trigger them, disable `merge_composites_with_shared_inputs` (and the follow-up MergeOptimizer) before compiling.

The `mode.excluding('fusion')` and `mode.excluding('inplace')` modes isolated which optimisation pass introduced the duplication:

```python
from pytensor.compile.mode import get_default_mode

for tag in ['fusion', 'inplace', ('fusion', 'inplace')]:
    excl = (tag,) if isinstance(tag, str) else tag
    fn = pytensor.function([sigma, ls, X, y], joint,
                           mode=get_default_mode().excluding(*excl))
    print(f'-{",".join(excl)}: {count_compiled(fn)}')
```

Result on this graph:

```
-fusion:           Cholesky=1, MatrixInverse=1
-inplace:          Cholesky=2, MatrixInverse=2     ← inplace alone doesn't dedup; the duplication exists pre-inplace
-fusion,inplace:   Cholesky=1, MatrixInverse=1
```

So fusion creates the duplication; inplace cannot dedup it (the duplicates exist before inplace runs); inplace then *prevents* any later dedup by assigning different `inplace_pattern`s.

To see *what's different* between the two `Composite` `Apply` nodes:

```python
from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar.basic import Composite

fn = pytensor.function([sigma, ls, X, y], joint)
groups = collections.defaultdict(list)
for nd in fn.maker.fgraph.apply_nodes:
    if isinstance(nd.op, Elemwise) and isinstance(nd.op.scalar_op, Composite):
        groups[tuple(id(i) for i in nd.inputs)].append(nd)

# Find groups of >1 Composite Apply with id-identical inputs
for inputs_key, applies in groups.items():
    if len(applies) > 1:
        for nd in applies:
            print(f'inputs id-tuple {inputs_key[:2]}...  '
                  f'inner outputs: {len(nd.op.scalar_op.outputs)}  '
                  f'op str: {nd.op.scalar_op}')
```

Output (shortened):

```
inputs id-tuple (4801000144, 4795399376)...  inner outputs: 1   Composite{exp((-0.5 * Maximum(...)))}
inputs id-tuple (4801000144, 4795399376)...  inner outputs: 3   Composite{...}
```

To dump the inner scalar fgraphs:

```python
import io
for inputs_key, applies in groups.items():
    if len(applies) > 1:
        for i, nd in enumerate(applies):
            buf = io.StringIO()
            pytensor.dprint(list(nd.op.scalar_op.outputs), file=buf)
            print(f'\n=== Composite[{i}] inner fgraph ===')
            print(buf.getvalue())
        break
```

This is what showed the gradient Composite has 3 outputs (`exp`, `Maximum`, `add`) while the forward has 1 (`exp`).

### A.6 Manual Composite-merge prototype (the proof-of-concept for Blocker A's fix)

```python
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import MergeOptimizer
from pytensor.graph.fg import FunctionGraph

# Find one mergeable group (siblings with id-identical inputs)
fn = pytensor.function([sigma, ls, X, y], joint,
                       mode=get_default_mode().excluding('inplace'))  # avoid the inplace blocker
fg = fn.maker.fgraph

groups = collections.defaultdict(list)
for nd in fg.apply_nodes:
    if isinstance(nd.op, Elemwise) and isinstance(nd.op.scalar_op, Composite):
        groups[tuple(id(i) for i in nd.inputs)].append(nd)
a, b = next(g for g in groups.values() if len(g) == 2)
ca, cb = a.op.scalar_op, b.op.scalar_op

# Step 1: build merged inner outputs by remapping cb's NominalScalarVariables onto ca's
remap = dict(zip(cb.inputs, ca.inputs))
b_outs_remapped = clone_replace(list(cb.outputs), remap)
all_inner = list(ca.outputs) + b_outs_remapped

# Step 2: inner CSE
temp_fg = FunctionGraph(inputs=list(ca.inputs), outputs=all_inner, clone=True)
mo = MergeOptimizer(); mo.add_requirements(temp_fg); mo.apply(temp_fg)

# Step 3: dedup output slots that now point to the same inner Variable
unique_outputs, slot_remap, seen = [], {}, {}
for i, o in enumerate(temp_fg.outputs):
    if id(o) in seen:
        slot_remap[i] = seen[id(o)]
    else:
        slot_remap[i] = len(unique_outputs)
        seen[id(o)] = len(unique_outputs)
        unique_outputs.append(o)

# Step 4: build the merged Composite and splice into the outer fgraph
merged = Composite(inputs=list(temp_fg.inputs), outputs=unique_outputs)
new = Elemwise(scalar_op=merged)(*a.inputs, return_list=True)
n_a = len(a.outputs)
repls = (
    list(zip(a.outputs, [new[slot_remap[i]] for i in range(n_a)])) +
    list(zip(b.outputs, [new[slot_remap[n_a + j]] for j in range(len(b.outputs))]))
)
fg.replace_all(repls, reason='manual-composite-merge')

# Step 5: outer MergeOptimizer collapses the now-equivalent SpecAssum/Add/Cholesky chains
mo2 = MergeOptimizer(); mo2.add_requirements(fg); mo2.apply(fg)

print(count_compiled(fn))   # Cholesky=1, MatrixInverse=1 — when run with mode.excluding('inplace')
```

Numerical sanity check: compile a fresh reference and compare:

```python
import numpy as np
fn_ref = pytensor.function([sigma, ls, X, y], joint)
rng = np.random.default_rng(0); N = 12
Xv = rng.standard_normal((N, 1)); yv = rng.standard_normal(N)
ref, got = fn_ref(0.5, 1.2, Xv, yv), fn(0.5, 1.2, Xv, yv)
for r, g in zip(ref, got):
    assert np.allclose(r, g, atol=1e-12)
```

### A.7 Blocker-B diagnostic (now no-op post-fix)

Identify the `MatrixInverse` Apply nodes and what's downstream of each. With `matrix_inverse_specialize` landed, this loop now finds nothing on the joint graph; the diagnostic is preserved here as the snippet that originally identified the blocker.

```python
from pytensor.tensor.linalg.inverse import MatrixInverse

fn = pytensor.function([sigma, ls, X, y], joint)
for n in fn.maker.fgraph.apply_nodes:
    op = n.op; core = op.core_op if isinstance(op, Blockwise) else op
    if isinstance(core, MatrixInverse):
        K = n.inputs[0]
        cli_ops = [type(c.op).__name__ for c, _ in fn.maker.fgraph.clients[n.outputs[0]]]
        print(f'MatrixInverse(K={type(K.owner.op).__name__})  consumers: {cli_ops}')
```

Pre-fix output (preserved for context):

```
MatrixInverse(K=Elemwise)  consumers: ['Elemwise']      ← contracted via a Composite for b·K⁻¹·dK/dθ·K⁻¹·b
MatrixInverse(K=Elemwise)  consumers: ['ExtractDiag']   ← diagonal pulled out for tr(K⁻¹·dK/dθ) when dK/dθ is diagonal (σ-grad)
```

To re-trigger the diagnostic against the unrewritten graph, disable the rule by stripping `matrix_inverse_specialize` from the specialize database (or run with `import ptgp` skipped, which leaves the assumption rules off and the standalone `MatrixInverse` Apply nodes intact).

---

## Appendix B — quick-reference: what each ptgp rule earns

`ptgp.rewrites` adds the following rules to PyTensor's optdb. Together they take the joint cubic-op counts to the floor for all three GP objectives (`Unapproximated` = 1, `VFE` = 2, `SVGP` = 1).

### Assumption rules (extend PyTensor's PSD inference)

| rule | what it does |
|---|---|
| `pt.assume(positive=True)` patch + `POSITIVE` propagation (`_elemwise_positive`, `_dimshuffle_positive`, `_alloc_positive`, `_extractdiag_positive`) | tags `σ` positive; lets `σ²·I` be recognised as PSD via `AllocDiag(POSITIVE)` etc. Bridge between "positive scalar" and "PSD matrix". |
| `_dot_xt_M_x_psd`, `_dot_xt_solve_x_psd`, `_dot_xt_chosolve_x_psd` | three small rules recognising `X.T @ M @ X`, `X.T @ Solve(M, X)`, `X.T @ CholeskySolve(L, X)` as PSD when `M`/`L` qualifies. Originally one combined `_dot_quadratic_form_psd` rule; split for clarity. |
| `_alloc_diag_psd_symbolic`, `_set_subtensor_psd` | `AllocDiag(positive vec)` and `set_subtensor(zeros, positive_vec, diag_idx)` recognised as PSD. Extends upstream's literal-constant PSD-AllocDiag rule to symbolic POSITIVE inputs. |
| `_mul_psd_with_symbolic_positive` | `c · A` PSD when `c` is symbolic POSITIVE and `A` is PSD. Extends upstream's literal-only Mul-PSD rule. |
| `_dimshuffle_psd` | matrix transpose of PSD is PSD (PSD ⇒ symmetric). |

### Structural rewrites

| rule | what it does | impact |
|---|---|---|
| `slogdet_specialize` (with `_existing_cholesky` reuse) | `SLogDet(A) → 2·sum(log diag L)` for PSD `A`, sharing an existing `Cholesky(A)`. Extended fast path for `A = L @ L.T` or `L.T @ L` (lower-triangular `L`): take `2·sum(log\|diag(L)\|)` directly, no Cholesky. | `SLogDet → 0` everywhere; in SVGP, also bypasses the wasteful Cholesky on `q_sqrt @ q_sqrt.T`. |
| `matrix_inverse_specialize` (with `_existing_cholesky` reuse) | `MatrixInverse(A) → cho_solve(L, eye)` for PSD `A`, sharing an existing `Cholesky(A)`. Extended fast path for `A = L @ L.T` or `L.T @ L`: use `L` directly via `cho_solve` or `solve_triangular` without computing a fresh Cholesky. | `MatrixInverse → 0` in joint graphs; closes Blockers B and most of E. |
| `det_of_LLT_to_diag_product` | `Det(L @ L.T)` or `Det(L.T @ L) → (prod(diag(L)))**2` for lower-triangular `L`. | Eliminates the standalone `Det` Apply that survives `slogdet_specialization` when `Det` is also referenced by the gradient pullback. |
| `diag_of_AAT_to_row_norms_squared` | `ExtractDiag(A @ A.T) → sum(A**2, axis=-1)` (and `A.T @ A → sum(A**2, axis=-2)`). Generic — no assumption needed. | Eliminates the M^2-element materialisation of `q_sqrt @ q_sqrt.T` that `pt.trace(q_cov)` would otherwise produce. Generally useful — not GP-specific. |
| `merge_composites_with_shared_inputs` + `merge_after_composite_dedup` (extra `MergeOptimizer` pass) | Finds sibling `Elemwise(Composite)` Apply nodes with id-identical inputs, merges them into one multi-output Composite with inner CSE, then re-runs `MergeOptimizer` to collapse the now-equivalent downstream chains. | Closes Blocker A on `Unapproximated`. Generic structural CSE; not GP-specific. |

### Compatibility patches

| patch | why |
|---|---|
| `_install_assume_patch` | upstream `pt.assume` doesn't accept `positive=`. Patch installs a wrapper that does, falling through to the upstream behaviour for everything else. Goes away once we upstream the POSITIVE assumption. |

No remaining open blockers for `Unapproximated`, `VFE`, or `SVGP`. All three at the cubic-op floor.
