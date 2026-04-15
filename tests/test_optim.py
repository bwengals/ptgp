"""Tests for ptgp.optim — native PyTensor training and prediction."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg


@pytest.fixture
def gp_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 40))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
    return X, y


@pytest.fixture
def svgp_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 80))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 80)
    return X, y


def test_compile_training_step_gp(gp_data):
    """GP trains and loss decreases."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.Matern52(ls=ls)
        gp = pg.GP(kernel=kernel, likelihood=pg.Gaussian(sigma=sigma))

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.compile_training_step(
        pg.marginal_log_likelihood, gp, X_var, y_var,
        pm_model=model, learning_rate=1e-2,
    )

    losses = []
    for i in range(100):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "Loss should decrease during training"


def test_compile_predict_gp(gp_data):
    """GP prediction returns reasonable shapes and values after training."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.Matern52(ls=ls)
        gp = pg.GP(kernel=kernel, likelihood=pg.Gaussian(sigma=sigma))

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.compile_training_step(
        pg.marginal_log_likelihood, gp, X_var, y_var,
        pm_model=model, learning_rate=1e-2,
    )

    for _ in range(200):
        train_step(X, y)

    X_new_var = pt.matrix("X_new")
    predict_fn = pg.compile_predict(
        gp, X_new_var, model, shared_params,
        X_train=X, y_train=y,
    )

    X_test = np.linspace(0, 5, 20)[:, None]
    fmean, fvar = predict_fn(X_test)

    assert fmean.shape == (20,)
    assert fvar.shape == (20,)
    assert np.all(fvar >= 0), "Predictive variance should be non-negative"
    # Predictions should roughly follow sin(x) after training
    assert np.corrcoef(fmean, np.sin(X_test[:, 0]))[0, 1] > 0.9


def test_compile_training_step_svgp(svgp_data):
    """SVGP trains with variational parameters as extra_vars."""
    X, y = svgp_data
    M = 15

    rng = np.random.default_rng(0)
    Z_init = rng.choice(X[:, 0], M, replace=False)
    Z_init = np.sort(Z_init)[:, None]

    q_mu_var = pt.vector("q_mu")
    q_sqrt_var = pt.matrix("q_sqrt")

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)

        kernel = eta**2 * pg.Matern52(ls=ls)
        svgp = pg.SVGP(
            kernel=kernel,
            likelihood=pg.Gaussian(sigma=0.1),
            inducing_variable=pg.InducingPoints(pt.as_tensor_variable(Z_init)),
            q_mu=q_mu_var,
            q_sqrt=q_sqrt_var,
        )

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.compile_training_step(
        pg.elbo, svgp, X_var, y_var,
        pm_model=model,
        extra_vars=[q_mu_var, q_sqrt_var],
        extra_init=[np.zeros(M), np.eye(M)],
        learning_rate=1e-2,
    )

    losses = []
    for i in range(100):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SVGP loss should decrease during training"


def test_sgd_optimizer(gp_data):
    """SGD optimizer works as alternative to adam."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.Matern52(ls=ls)
        gp = pg.GP(kernel=kernel, likelihood=pg.Gaussian(sigma=sigma))

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.compile_training_step(
        pg.marginal_log_likelihood, gp, X_var, y_var,
        pm_model=model,
        optimizer_fn=pg.sgd,
        learning_rate=1e-3,
    )

    losses = []
    for i in range(50):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SGD should also reduce loss"
