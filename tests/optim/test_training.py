"""Tests for ptgp.optim.training — native PyTensor training and prediction."""

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

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        learning_rate=1e-2,
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

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        learning_rate=1e-2,
    )

    for _ in range(200):
        train_step(X, y)

    X_new_var = pt.matrix("X_new")
    predict_fn = pg.optim.compile_predict(
        gp,
        X_new_var,
        model,
        shared_params,
        X_train=X,
        y_train=y,
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

    vp = pg.gp.init_variational_params(M)

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
            variational_params=vp,
        )

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )

    losses = []
    for i in range(100):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SVGP loss should decrease during training"


def test_prior_shifts_optimum(gp_data):
    """Training with include_prior=True converges to a different point
    than include_prior=False when the prior is strong enough to pull
    the optimum away from the MLE.
    """
    X, y = gp_data

    def build():
        with pm.Model() as model:
            # Tight prior on ls, far from whatever the MLE would pick.
            ls = pm.Normal("ls", mu=5.0, sigma=0.01)
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.Exponential("sigma", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)
        return model, gp

    def train(include_prior):
        model, gp = build()
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        train_step, shared_params, _ = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
            learning_rate=1e-2,
            include_prior=include_prior,
        )
        for _ in range(500):
            train_step(X, y)
        return pg.optim.get_trained_params(model, shared_params)

    mle = train(include_prior=False)
    map_ = train(include_prior=True)

    # The tight prior should pin ls near 5.0 in constrained space.
    assert abs(map_["ls"] - 5.0) < 0.1
    # The MLE is free to wander far from the prior mean.
    assert abs(mle["ls"] - 5.0) > 0.5


def test_sgd_optimizer(gp_data):
    """SGD optimizer works as alternative to adam."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        optimizer_fn=pg.optim.sgd,
        learning_rate=1e-3,
    )

    losses = []
    for i in range(50):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SGD should also reduce loss"
