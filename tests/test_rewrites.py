"""Tests for the assumption rules and rewrites added in ``ptgp.rewrites``.

Written so that, when the rules are upstreamed into PyTensor, the test bodies
move over unchanged — only the imports of ``POSITIVE`` (and the side-effect
import that installs the rules) would shift from ``ptgp`` to ``pytensor``.
"""

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.assumptions import (
    AssumptionFeature,
    POSITIVE_DEFINITE,
    SYMMETRIC,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.summary import SLogDet

# Install the assumption rules and rewrites under test (side-effect import).
import ptgp.rewrites  # noqa: F401
from ptgp.rewrites import POSITIVE


def make_fgraph(*outputs, inputs=None):
    fg = FunctionGraph(outputs=list(outputs), inputs=inputs, clone=False)
    af = AssumptionFeature()
    fg.attach_feature(af)
    return fg, af


# ---------------------------------------------------------------------------
# POSITIVE assumption propagation
# ---------------------------------------------------------------------------


def test_specify_assumptions_accepts_positive_kwarg():
    x = pt.dscalar("x")
    y = pt.specify_assumptions(x, positive=True)
    _, af = make_fgraph(y, inputs=[x])
    assert af.check(y, POSITIVE)


def test_positive_propagates_through_sqr_and_pow():
    x = pt.specify_assumptions(pt.dscalar("x"), positive=True)
    _, af = make_fgraph(x**2, x**3.5, inputs=[x.owner.inputs[0]])
    assert af.check(x**2, POSITIVE)
    assert af.check(x**3.5, POSITIVE)


def test_positive_propagates_through_mul_only_if_all_positive():
    x = pt.specify_assumptions(pt.dscalar("x"), positive=True)
    y = pt.specify_assumptions(pt.dscalar("y"), positive=True)
    z = pt.dscalar("z")  # no assumption
    _, af = make_fgraph(x * y, x * z, inputs=[x.owner.inputs[0], y.owner.inputs[0], z])
    assert af.check(x * y, POSITIVE)
    assert not af.check(x * z, POSITIVE)


def test_positive_propagates_through_dimshuffle():
    x = pt.specify_assumptions(pt.dvector("x"), positive=True)
    _, af = make_fgraph(x[None, :], inputs=[x.owner.inputs[0]])
    assert af.check(x[None, :], POSITIVE)


def test_alloc_of_positive_constant_is_positive():
    n = pt.lscalar("n")
    ones = pt.ones((n,))
    _, af = make_fgraph(ones, inputs=[n])
    assert af.check(ones, POSITIVE)


def test_alloc_of_zero_is_not_positive():
    n = pt.lscalar("n")
    zeros = pt.zeros((n,))
    _, af = make_fgraph(zeros, inputs=[n])
    assert not af.check(zeros, POSITIVE)


def test_diagonal_of_identity_is_positive():
    n = pt.lscalar("n")
    diag_eye = pt.diagonal(pt.eye(n))
    _, af = make_fgraph(diag_eye, inputs=[n])
    assert af.check(diag_eye, POSITIVE)


# ---------------------------------------------------------------------------
# POSITIVE_DEFINITE inference for new patterns
# ---------------------------------------------------------------------------


def test_alloc_diag_of_symbolic_positive_vector_is_psd():
    v = pt.specify_assumptions(pt.dvector("v"), positive=True)
    M = pt.diag(v)
    _, af = make_fgraph(M, inputs=[v.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


def test_alloc_diag_of_unknown_vector_is_not_psd():
    v = pt.dvector("v")
    M = pt.diag(v)
    _, af = make_fgraph(M, inputs=[v])
    assert not af.check(M, POSITIVE_DEFINITE)


def test_mul_of_positive_scalar_and_psd_matrix_is_psd():
    sigma = pt.specify_assumptions(pt.dscalar("sigma"), positive=True)
    K = pt.specify_assumptions(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    M = sigma**2 * K
    _, af = make_fgraph(M, inputs=[sigma.owner.inputs[0], K.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


def test_psd_propagates_through_matrix_transpose():
    K = pt.specify_assumptions(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, af = make_fgraph(K.T, inputs=[K.owner.inputs[0]])
    assert af.check(K.T, POSITIVE_DEFINITE)


def test_quadratic_form_dot_is_psd():
    """X.T @ M @ X is PSD when M is PSD."""
    X = pt.dmatrix("X")
    M = pt.specify_assumptions(pt.dmatrix("M"), positive_definite=True, symmetric=True)
    Q = X.T.dot(M.dot(X))
    _, af = make_fgraph(Q, inputs=[X, M.owner.inputs[0]])
    assert af.check(Q, POSITIVE_DEFINITE)


def test_quadratic_form_with_solve_is_psd():
    """X.T @ M^{-1} @ X (written as Solve) is PSD when M is PSD."""
    X = pt.dmatrix("X")
    M = pt.specify_assumptions(pt.dmatrix("M"), positive_definite=True, symmetric=True)
    Q = X.T.dot(pt.linalg.solve(M, X))
    _, af = make_fgraph(Q, inputs=[X, M.owner.inputs[0]])
    assert af.check(Q, POSITIVE_DEFINITE)


def test_set_subtensor_of_zeros_with_positive_diagonal_is_psd():
    """``zeros(N, N)[arange(N), arange(N)] = positive_vec`` is PSD."""
    n = pt.lscalar("n")
    v = pt.specify_assumptions(pt.dvector("v"), positive=True)
    base = pt.zeros((n, n))
    idx = pt.arange(n)
    M = pt.set_subtensor(base[idx, idx], v)
    _, af = make_fgraph(M, inputs=[n, v.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


# ---------------------------------------------------------------------------
# SLogDet -> Cholesky rewrite
# ---------------------------------------------------------------------------


def _has_op(graph, op_type):
    fg = FunctionGraph(outputs=[graph] if not isinstance(graph, list) else graph, clone=False)
    for node in fg.apply_nodes:
        op = node.op
        core = op.core_op if isinstance(op, Blockwise) else op
        if isinstance(core, op_type):
            return True
    return False


def test_slogdet_of_psd_is_lowered_to_cholesky():
    K = pt.specify_assumptions(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    assert not _has_op(rewritten, SLogDet)
    assert _has_op(rewritten, Cholesky)


def test_slogdet_without_psd_is_not_lowered():
    K = pt.dmatrix("K")  # no PSD annotation
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    # Without the PSD assumption the Cholesky-based rewrite must not fire.
    assert not _has_op(rewritten, Cholesky)


def test_slogdet_reuses_existing_cholesky():
    """When an upstream Solve has already produced ``Cholesky(K)``, SLogDet should reuse it."""
    K = pt.specify_assumptions(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    b = pt.dvector("b")
    x = pt.linalg.solve(K, b, assume_a="pos")
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph([x, logdet], include=("fast_run",))
    fg = FunctionGraph(outputs=rewritten, clone=False)
    n_chol = sum(
        1
        for node in fg.apply_nodes
        if isinstance(node.op.core_op if isinstance(node.op, Blockwise) else node.op, Cholesky)
    )
    assert n_chol == 1, f"expected SLogDet to share Cholesky, got {n_chol} factorisations"


def test_slogdet_lowering_is_numerically_correct():
    K = pt.specify_assumptions(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, logdet = pt.linalg.slogdet(K)
    f = function([K.owner.inputs[0]], logdet)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    K_val = A @ A.T + np.eye(6)
    np.testing.assert_allclose(f(K_val), np.linalg.slogdet(K_val)[1], atol=1e-8)
