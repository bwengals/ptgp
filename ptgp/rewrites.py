"""PTGP-local PyTensor rewrites and assumption rules.

Registered into PyTensor's global registries at import time. See REWRITES.md.
"""

import pytensor.tensor as pt
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import (
    MergeOptimizer,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from pytensor.scalar.basic import Composite, Mul, Pow, Sqr
from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.diagonal import indexes_diagonal
from pytensor.tensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.tensor.assumptions.specify import SpecifyAssumptions
from pytensor.tensor.assumptions.utils import check_assumption, eye_is_identity, true_if
from pytensor.tensor.basic import (
    Alloc,
    AllocDiag,
    Eye,
    ExtractDiag,
    NotScalarConstantError,
    as_tensor_variable,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.psd import cho_solve
from pytensor.tensor.linalg.summary import SLogDet
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.subtensor import AdvancedIncSubtensor, IncSubtensor


# ---------------------------------------------------------------------------
# POSITIVE assumption: a (real) tensor whose every element is strictly > 0.
# ---------------------------------------------------------------------------

POSITIVE = AssumptionKey("positive")


def _specify_assumptions(
    x,
    diagonal=None,
    lower_triangular=None,
    upper_triangular=None,
    symmetric=False,
    positive_definite=False,
    orthogonal=False,
    positive=False,
):
    """Drop-in replacement for ``pt.specify_assumptions`` that also accepts ``positive=True``."""
    x = as_tensor_variable(x)
    names = {
        name
        for name, value in [
            ("diagonal", diagonal),
            ("lower_triangular", lower_triangular),
            ("upper_triangular", upper_triangular),
            ("symmetric", symmetric),
            ("positive_definite", positive_definite),
            ("orthogonal", orthogonal),
            ("positive", positive),
        ]
        if value
    }
    if not names:
        return x
    return SpecifyAssumptions(frozenset(names))(x)


# Patch every public binding of ``specify_assumptions`` so ``pt.specify_assumptions(sigma, positive=True)``
# works regardless of which module the caller imported it from.
def _install_specify_patch():
    import pytensor.tensor as _pt
    import pytensor.tensor.assumptions as _assumptions_pkg
    from pytensor.tensor.assumptions import specify as _specify_module

    for module in (_pt, _assumptions_pkg, _specify_module):
        module.specify_assumptions = _specify_assumptions


_install_specify_patch()


@register_assumption(POSITIVE, Elemwise)
def _elemwise_positive(op, feature, fgraph, node, input_states):
    """Propagate POSITIVE through a few elementwise ops.

    - ``Sqr(x)`` is positive iff ``x`` is positive (strict; rules out x == 0).
    - ``Mul(a, b, ...)`` is positive iff every input is positive.
    """
    scalar_op = op.scalar_op
    if isinstance(scalar_op, Sqr):
        return true_if(bool(input_states[0]))
    if isinstance(scalar_op, Pow):
        # ``positive_base ** real_exponent`` is strictly positive for any real exponent.
        return true_if(bool(input_states[0]))
    if isinstance(scalar_op, Mul):
        return true_if(all(bool(s) for s in input_states))
    return [FactState.UNKNOWN] * len(node.outputs)


@register_assumption(POSITIVE, DimShuffle)
def _dimshuffle_positive(op, feature, fgraph, node, input_states):
    """POSITIVE survives reshape/broadcast: every element of the input is also an element of the output."""
    return true_if(bool(input_states[0]))


@register_assumption(POSITIVE_DEFINITE, DimShuffle)
def _dimshuffle_psd(op, feature, fgraph, node, input_states):
    """``A.T`` is PSD when ``A`` is PSD (PSD ⇒ symmetric)."""
    if op.is_matrix_transpose:
        return true_if(bool(input_states[0]))
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE, Alloc)
def _alloc_positive(op, feature, fgraph, node, input_states):
    """``Alloc(c, *shape)`` is positive iff the fill value ``c`` is."""
    fill = node.inputs[0]
    if bool(input_states[0]):
        return [FactState.TRUE]
    try:
        if get_underlying_scalar_constant_value(fill) > 0:
            return [FactState.TRUE]
    except NotScalarConstantError:
        pass
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE, ExtractDiag)
def _extractdiag_positive(op, feature, fgraph, node, input_states):
    """``diag(Eye(N))`` is the all-ones vector — strictly positive."""
    if op.offset != 0:
        return [FactState.UNKNOWN]
    [m] = node.inputs
    owner = m.owner
    if owner is not None and isinstance(owner.op, Eye) and eye_is_identity(owner):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


# ---------------------------------------------------------------------------
# Extend POSITIVE_DEFINITE Mul rule to accept symbolic POSITIVE scalars.
# Upstream only handles literal-constant positive scalars; this rule fires
# afterwards when the constant check fails.
# ---------------------------------------------------------------------------


def _is_zero_alloc(var):
    """True when *var* is an ``Alloc`` whose fill value is the constant 0."""
    owner = var.owner
    if owner is None or not isinstance(owner.op, Alloc):
        return False
    fill = owner.inputs[0]
    if not isinstance(fill, Constant):
        return False
    try:
        return float(fill.data) == 0.0
    except (TypeError, ValueError):
        return False


def _matrix_transpose_of(var):
    """Return the underlying ``X`` if ``var`` is ``X.T``, else None."""
    owner = var.owner
    if owner is None:
        return None
    if isinstance(owner.op, DimShuffle) and owner.op.is_matrix_transpose:
        return owner.inputs[0]
    return None


def _solve_core_op(var):
    """Return the core ``Solve`` op if ``var`` is a (Blockwise-wrapped) Solve, else None."""
    owner = var.owner
    if owner is None:
        return None
    op = owner.op
    core = op.core_op if isinstance(op, Blockwise) else op
    return core if isinstance(core, Solve) else None


@register_assumption(POSITIVE_DEFINITE, Dot)
def _dot_quadratic_form_psd(op, feature, fgraph, node, input_states):
    """``X.T @ M @ X`` and ``X.T @ Solve(M, X)`` are PSD when ``M`` is PSD."""
    a, b = node.inputs
    X = _matrix_transpose_of(a)
    if X is None:
        return [FactState.UNKNOWN]
    # Right-associative explicit form: X.T @ (M @ X) where M is PSD
    if b.owner is not None and isinstance(b.owner.op, Dot):
        M, X2 = b.owner.inputs
        if X2 is X and feature.check(M, POSITIVE_DEFINITE):
            return [FactState.TRUE]
    # Solve canonical form: X.T @ Solve(M, X) ≡ X.T @ M^{-1} @ X
    if _solve_core_op(b) is not None:
        M, X2 = b.owner.inputs
        if X2 is X and feature.check(M, POSITIVE_DEFINITE):
            return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, AllocDiag)
def _alloc_diag_psd_symbolic(op, feature, fgraph, node, input_states):
    """``AllocDiag(positive_vector)`` → PSD (extends upstream rule beyond literal constants)."""
    if op.offset != 0:
        return [FactState.UNKNOWN]
    [diag_values] = node.inputs
    if feature.check(diag_values, POSITIVE):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, AdvancedIncSubtensor)
@register_assumption(POSITIVE_DEFINITE, IncSubtensor)
def _set_subtensor_psd(op, feature, fgraph, node, input_states):
    """``set_subtensor(zeros, positive_values, diag, diag)`` → PSD diagonal matrix."""
    if not getattr(op, "set_instead_of_inc", False):
        return [FactState.UNKNOWN]
    if not indexes_diagonal(node):
        return [FactState.UNKNOWN]
    base, values = node.inputs[0], node.inputs[1]
    if not _is_zero_alloc(base):
        return [FactState.UNKNOWN]
    if not feature.check(values, POSITIVE):
        return [FactState.UNKNOWN]
    return [FactState.TRUE]


@register_assumption(POSITIVE_DEFINITE, Elemwise)
def _mul_psd_with_symbolic_positive(op, feature, fgraph, node, input_states):
    """``c * A`` is PSD when ``c`` is a POSITIVE scalar (possibly symbolic) and ``A`` is PSD."""
    if not isinstance(op.scalar_op, Mul):
        return [FactState.UNKNOWN] * len(node.outputs)

    inputs = node.inputs
    for i, inp in enumerate(inputs):
        if not feature.check(inp, POSITIVE):
            continue
        others = [inputs[j] for j in range(len(inputs)) if j != i]
        if all(feature.check(o, POSITIVE_DEFINITE) for o in others):
            return [FactState.TRUE]
    return [FactState.UNKNOWN] * len(node.outputs)


# ---------------------------------------------------------------------------
# SLogDet(PSD A) -> (1, 2 * sum(log(diag(cholesky(A))))).
# ---------------------------------------------------------------------------


def _existing_cholesky(fgraph, A):
    """Return an existing ``Cholesky(lower=True)(A)`` output already in *fgraph*, else None.

    Lets SLogDet share a factor produced by an upstream Solve lowering instead
    of computing a second one.
    """
    for client, _ in fgraph.clients.get(A, ()):
        op = client.op
        core_op = op.core_op if isinstance(op, Blockwise) else op
        if isinstance(core_op, Cholesky) and core_op.lower:
            return client.outputs[0]
    return None


@register_specialize
@node_rewriter([SLogDet])
def slogdet_psd_to_cholesky(fgraph, node):
    """SLogDet(A) -> (1, 2*sum(log(diag(cholesky(A))))) for PSD A."""
    [A] = node.inputs
    if not check_assumption(fgraph, A, POSITIVE_DEFINITE):
        return None

    L = _existing_cholesky(fgraph, A)
    if L is None:
        L = cholesky(A, lower=True)

    sign_old, logabsdet_old = node.outputs
    sign_new = pt.ones((), dtype=sign_old.dtype)
    logabsdet_new = (2.0 * pt.log(pt.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)).astype(
        logabsdet_old.dtype
    )
    copy_stack_trace([sign_old, logabsdet_old], [sign_new, logabsdet_new])
    return [sign_new, logabsdet_new]


# ---------------------------------------------------------------------------
# MatrixInverse(PSD A) -> cho_solve(L, eye), reusing an existing Cholesky if
# present. Avoids the redundant cubic factorisation that pt.grad(slogdet)
# triggers via its standalone MatrixInverse cotangent.
# ---------------------------------------------------------------------------


@register_specialize
@node_rewriter([blockwise_of(MatrixInverse)])
def matrix_inverse_psd_to_cholesky(fgraph, node):
    """MatrixInverse(A) -> cho_solve(L, eye) for PSD A, reusing an existing Cholesky(A)."""
    [A] = node.inputs
    if not check_assumption(fgraph, A, POSITIVE_DEFINITE):
        return None

    L = _existing_cholesky(fgraph, A)
    if L is None:
        L = cholesky(A, lower=True)

    eye = pt.eye(A.shape[-1], dtype=A.dtype)
    inv_A = cho_solve((L, True), eye)
    copy_stack_trace(node.outputs[0], inv_A)
    return [inv_A]


# ---------------------------------------------------------------------------
# Merge sibling Elemwise(Composite) Apply nodes that consume the exact same
# inputs but expose different output sets. FusionOptimizer's greedy convex
# closure produces this pattern when forward and gradient consumers of a
# shared Elemwise sub-expression sit in different fuseable contexts; the
# duplication then propagates to downstream Cholesky / MatrixInverse Apply
# nodes that can't be CSE'd because they end up with different
# inplace_patterns. Has to run after FusionOptimizer (top-level position 49)
# and before InplaceElemwiseOptimizer (top-level position 50.5). See
# REWRITE_ANALYSIS.md §5 for the diagnosis.
# ---------------------------------------------------------------------------


@node_rewriter([Elemwise])
def merge_composites_with_shared_inputs(fgraph, node):
    """Fuse two Elemwise(Composite) Apply nodes with id-identical inputs into one multi-output Composite."""
    if not isinstance(node.op.scalar_op, Composite):
        return None
    inputs = tuple(node.inputs)
    if not inputs:
        return None

    sibling = None
    for client, _ in fgraph.clients[inputs[0]]:
        if client is node:
            continue
        if not (isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, Composite)):
            continue
        if tuple(client.inputs) != inputs:
            continue
        sibling = client
        break
    if sibling is None:
        return None

    ca, cb = node.op.scalar_op, sibling.op.scalar_op
    remap = dict(zip(cb.inputs, ca.inputs))
    cb_outs_remapped = clone_replace(list(cb.outputs), remap)

    # Inner CSE so the merged Composite computes the shared sub-expression once.
    temp_fg = FunctionGraph(
        inputs=list(ca.inputs),
        outputs=list(ca.outputs) + cb_outs_remapped,
        clone=True,
    )
    inner_merge = MergeOptimizer()
    inner_merge.add_requirements(temp_fg)
    inner_merge.apply(temp_fg)

    # Output-slot dedup: any two slots pointing to the same inner Variable
    # collapse to one external slot.
    unique_outputs, slot_remap, seen = [], {}, {}
    for i, o in enumerate(temp_fg.outputs):
        if id(o) in seen:
            slot_remap[i] = seen[id(o)]
        else:
            slot_remap[i] = len(unique_outputs)
            seen[id(o)] = len(unique_outputs)
            unique_outputs.append(o)

    merged = Composite(inputs=list(temp_fg.inputs), outputs=unique_outputs)
    new_outs = Elemwise(scalar_op=merged)(*node.inputs, return_list=True)

    n_a = len(node.outputs)
    replacements = {}
    for i, out in enumerate(node.outputs):
        replacements[out] = new_outs[slot_remap[i]]
    for j, out in enumerate(sibling.outputs):
        replacements[out] = new_outs[slot_remap[n_a + j]]
    copy_stack_trace(list(node.outputs) + list(sibling.outputs), list(replacements.values()))
    return replacements


# Register at top-level optdb position 49.7: after add_destroy_handler (49.5)
# and before any inplace pass (earliest is blockwise_inplace at 50.10).
optdb.register(
    "merge_composites_with_shared_inputs",
    in2out(merge_composites_with_shared_inputs, ignore_newtrees=True),
    "fast_run",
    position=49.7,
)

# After the Composite merge above, downstream SpecifyAssumptions / Add /
# Cholesky / MatrixInverse Apply nodes that were previously distinct now
# have id-identical inputs and should CSE into one Apply. Pytensor's
# stock pipeline only runs MergeOptimizer at position 49 (before fusion),
# so without this second pass the duplicates persist into inplace
# optimization, which then assigns them different inplace_patterns and
# permanently locks in the duplication.
optdb.register(
    "merge_after_composite_dedup",
    MergeOptimizer(),
    "fast_run",
    position=49.75,
)
