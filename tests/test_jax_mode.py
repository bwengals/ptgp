"""Smoke tests for JAX compilation mode on VFE and SVGP training+predict paths."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg
from ptgp.gp.svgp import SVGP, VariationalParams
from ptgp.gp.vfe import VFE
from ptgp.inducing import Points
from ptgp.kernels.stationary import Matern32
from ptgp.likelihoods.gaussian import Gaussian
from ptgp.objectives import collapsed_elbo, elbo
from ptgp.optim.training import compile_predict, compile_training_step

JAX_KWARGS = {"compile_kwargs": {"mode": "JAX"}}


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    N, M = 40, 8
    X = np.sort(rng.uniform(0, 1, N))[:, None]
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(N)
    Z = np.linspace(0, 1, M)[:, None]
    return X, y, Z, N, M


def test_vfe_jax_train_step_decreases_loss(data):
    X, y, Z, N, M = data
    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * Matern32(input_dim=1, ls=ls)
        gp = VFE(kernel=kernel, sigma=0.1, inducing_variable=Points(Z))

    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y", shape=(None,))
    step, _, _ = compile_training_step(
        lambda gp, X, y: collapsed_elbo(gp, X, y).elbo,
        gp, X_var, y_var, model=model, learning_rate=1e-2, **JAX_KWARGS,
    )
    loss0 = float(step(X, y))
    for _ in range(20):
        step(X, y)
    loss1 = float(step(X, y))
    assert loss1 < loss0


def test_svgp_whitened_jax_train_and_predict(data):
    X, y, Z, N, M = data
    q_mu_var = pt.vector("q_mu", shape=(M,))
    q_sqrt_var = pt.matrix("q_sqrt", shape=(M, M))
    vp = VariationalParams(
        q_mu=q_mu_var,
        q_sqrt=pt.assume(q_sqrt_var, lower_triangular=True),
        extra_vars=[q_mu_var, q_sqrt_var],
        extra_init=[np.zeros(M), np.eye(M)],
    )
    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * Matern32(input_dim=1, ls=ls)
        svgp = SVGP(
            kernel=kernel, likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(Z), variational_params=vp, whiten=True,
        )

    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y", shape=(None,))
    step, sp, se = compile_training_step(
        lambda gp, X, y: elbo(gp, X, y).elbo,
        svgp, X_var, y_var, model=model,
        extra_vars=vp.extra_vars, extra_init=vp.extra_init,
        learning_rate=1e-2, **JAX_KWARGS,
    )
    loss0 = float(step(X, y))
    for _ in range(20):
        step(X, y)
    loss1 = float(step(X, y))
    assert loss1 < loss0

    X_new = pt.matrix("X_new", shape=(None, 1))
    pred = compile_predict(
        svgp, X_new, model, sp,
        extra_vars=vp.extra_vars, shared_extras=se,
        **JAX_KWARGS,
    )
    m, v = pred(X)
    assert m.shape == (N,) and v.shape == (N,)
    assert np.all(np.isfinite(m)) and np.all(v >= 0)


def test_svgp_unwhitened_jax_train_step_decreases_loss(data):
    X, y, Z, N, M = data
    vp = VariationalParams(
        q_mu=pt.as_tensor(np.zeros(M)),
        q_sqrt=pt.assume(pt.as_tensor(np.eye(M)), lower_triangular=True),
    )
    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * Matern32(input_dim=1, ls=ls)
        svgp = SVGP(
            kernel=kernel, likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(Z), variational_params=vp, whiten=False,
        )

    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y", shape=(None,))
    step, _, _ = compile_training_step(
        lambda gp, X, y: elbo(gp, X, y).elbo,
        svgp, X_var, y_var, model=model,
        learning_rate=1e-2, **JAX_KWARGS,
    )
    loss0 = float(step(X, y))
    for _ in range(20):
        step(X, y)
    loss1 = float(step(X, y))
    assert loss1 < loss0
