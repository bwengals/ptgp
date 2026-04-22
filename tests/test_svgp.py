"""SVGP tests for non-Gaussian likelihoods, cross-checked against GPJax.

The likelihood unit tests already verify the Gauss-Hermite
``variational_expectation`` in isolation. These tests close the remaining
gap: exercising the full SVGP ELBO wiring (predict + KL + variational
expectation) with a non-Gaussian likelihood end-to-end.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from scipy.special import erf

import gpjax as gpx

import ptgp as pg

ATOL = 1e-5


def _binary_data(rng, n=80):
    """1D Bernoulli data with boundary near x=0, labels derived from true latent."""
    X = np.sort(rng.uniform(-3, 3, n))[:, None]
    p = 0.5 * (1.0 + erf(X[:, 0] / np.sqrt(2.0)))
    y = (rng.uniform(0, 1, n) < p).astype(np.float64)
    return X, y


class TestSVGPBernoulliSmoke:
    """End-to-end sanity: SVGP + Bernoulli trains and recovers the class
    boundary. Catches everything between "graph compiles" and "trained
    model is useful" — gradient plumbing, whitening wiring, variational
    parameter optimization, predictive path. A failure here means the
    ELBO or its gradient is wrong in a way that breaks training, even
    if individual components look fine in isolation.
    """

    def test_loss_decreases_and_classifies(self):
        """Loss decreases over 400 Adam steps and final accuracy > 0.85."""
        rng = np.random.default_rng(0)
        X, y = _binary_data(rng, n=120)
        M = 12

        q_mu_var = pt.vector("q_mu")
        q_sqrt_var = pt.matrix("q_sqrt")
        Z_init = np.linspace(-3, 3, M)[:, None]

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            svgp = pg.gp.SVGP(
                kernel=kernel,
                likelihood=pg.likelihoods.Bernoulli(),
                inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
                q_mu=q_mu_var,
                q_sqrt=q_sqrt_var,
            )

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        train_step, shared_params, shared_extras = pg.optim.compile_training_step(
            pg.objectives.elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=[q_mu_var, q_sqrt_var],
            extra_init=[np.zeros(M), np.eye(M)],
            learning_rate=5e-2,
        )

        losses = [float(train_step(X, y)) for _ in range(400)]
        assert losses[-1] < losses[0], "SVGP+Bernoulli loss should decrease"

        X_new_var = pt.matrix("X_new")
        predict_fn = pg.optim.compile_predict(
            svgp,
            X_new_var,
            model,
            shared_params,
            extra_vars=[q_mu_var, q_sqrt_var],
            shared_extras=shared_extras,
            incl_lik=True,
        )
        p_mean, _ = predict_fn(X)
        y_pred = (p_mean > 0.5).astype(np.float64)
        acc = float(np.mean(y_pred == y))
        assert acc > 0.85, f"classification accuracy {acc:.2f} too low"


class TestSVGPBernoulliElboMatchesGPJax:
    """Evaluate the whitened-SVGP ELBO in PTGP and GPJax at a fixed
    configuration (hyperparameters, inducing points, q_mu, q_sqrt all
    identical) and require the two scalars to match at atol=1e-5. No
    optimizer — this pins the ELBO math (predict + KL + variational
    expectation) against a reference implementation.
    """

    def _fixed_config(self, rng, N=40, M=8):
        X = np.sort(rng.uniform(-3, 3, N))[:, None]
        y = rng.integers(0, 2, N).astype(np.float64)
        Z = np.linspace(-3, 3, M)[:, None]
        q_mu = rng.normal(0, 0.3, M)
        # Lower-triangular factor with positive diagonal.
        L = np.tril(rng.normal(0, 0.2, (M, M)))
        L[np.arange(M), np.arange(M)] = np.abs(L[np.arange(M), np.arange(M)]) + 0.5
        return X, y, Z, q_mu, L

    def _ptgp_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val):
        """Evaluate PTGP whitened-SVGP ELBO at the fixed configuration."""
        ls = pt.scalar("ls")
        eta = pt.scalar("eta")
        q_mu = pt.vector("q_mu")
        q_sqrt = pt.matrix("q_sqrt")
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Bernoulli(),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=True,
        )
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        elbo_expr = pg.objectives.elbo(svgp, X_var, y_var)
        fn = pytensor.function([X_var, y_var, q_mu, q_sqrt, ls, eta], elbo_expr)
        return float(fn(X, y, q_mu_val, q_sqrt_val, ls_val, eta_val))

    def _gpjax_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val):
        """Evaluate GPJax whitened-SVGP ELBO at the same configuration."""
        kernel = gpx.kernels.Matern52(
            active_dims=[0], lengthscale=jnp.array(ls_val), variance=jnp.array(eta_val**2)
        )
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        likelihood = gpx.likelihoods.Bernoulli(num_datapoints=X.shape[0])
        posterior = prior * likelihood
        # jitter=0 so the reference matches PTGP's no-jitter Kzz exactly.
        q = gpx.variational_families.WhitenedVariationalGaussian(
            posterior=posterior,
            inducing_inputs=jnp.array(Z),
            variational_mean=jnp.array(q_mu_val)[:, None],
            variational_root_covariance=jnp.array(q_sqrt_val),
            jitter=0.0,
        )
        data = gpx.Dataset(X=jnp.array(X), y=jnp.array(y)[:, None])
        return float(gpx.objectives.elbo(q, data))

    def test_elbo_match(self):
        rng = np.random.default_rng(1)
        X, y, Z, q_mu_val, q_sqrt_val = self._fixed_config(rng)
        ls_val, eta_val = 1.3, 0.9

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val)
        e_gpjax = self._gpjax_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val)

        np.testing.assert_allclose(e_ptgp, e_gpjax, atol=ATOL)


def _count_data(rng, n=80):
    """1D Poisson data with log-linear rate; `y` is integer counts."""
    X = np.sort(rng.uniform(-2, 2, n))[:, None]
    rate = np.exp(0.5 * X[:, 0] + 0.3)
    y = rng.poisson(rate).astype(np.float64)
    return X, y


class TestSVGPPoissonSmoke:
    """End-to-end sanity: SVGP + Poisson trains and the predicted rate
    tracks the true log-linear rate. Poisson exercises the closed-form
    variational expectation branch (log link), so a failure here points
    at the closed-form VE wiring or the SVGP predictive path — distinct
    from the Bernoulli smoke test, which exercises the quadrature branch.
    """

    def test_loss_decreases_and_rate_correlates(self):
        """Loss decreases over 400 Adam steps; predicted rate correlates
        with true rate at Pearson r > 0.8."""
        rng = np.random.default_rng(2)
        X, y = _count_data(rng, n=120)
        M = 12

        q_mu_var = pt.vector("q_mu")
        q_sqrt_var = pt.matrix("q_sqrt")
        Z_init = np.linspace(-2, 2, M)[:, None]

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            svgp = pg.gp.SVGP(
                kernel=kernel,
                likelihood=pg.likelihoods.Poisson(),
                inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
                q_mu=q_mu_var,
                q_sqrt=q_sqrt_var,
            )

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        train_step, shared_params, shared_extras = pg.optim.compile_training_step(
            pg.objectives.elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=[q_mu_var, q_sqrt_var],
            extra_init=[np.zeros(M), np.eye(M)],
            learning_rate=5e-2,
        )

        losses = [float(train_step(X, y)) for _ in range(400)]
        assert losses[-1] < losses[0], "SVGP+Poisson loss should decrease"

        X_new_var = pt.matrix("X_new")
        predict_fn = pg.optim.compile_predict(
            svgp,
            X_new_var,
            model,
            shared_params,
            extra_vars=[q_mu_var, q_sqrt_var],
            shared_extras=shared_extras,
            incl_lik=True,
        )
        rate_pred, _ = predict_fn(X)
        rate_true = np.exp(0.5 * X[:, 0] + 0.3)
        r = float(np.corrcoef(rate_pred, rate_true)[0, 1])
        assert r > 0.8, f"rate correlation {r:.2f} too low"


class TestSVGPPoissonElboMatchesGPJax:
    """Evaluate the whitened-SVGP ELBO in PTGP and GPJax at a fixed
    Poisson configuration and require the two scalars to match at
    atol=1e-5. Pins the closed-form Poisson variational expectation
    plus the rest of the ELBO (predict + KL) against GPJax, independent
    of any optimizer.
    """

    def _fixed_config(self, rng, N=40, M=8):
        X = np.sort(rng.uniform(-2, 2, N))[:, None]
        rate = np.exp(0.5 * X[:, 0] + 0.3)
        y = rng.poisson(rate).astype(np.float64)
        Z = np.linspace(-2, 2, M)[:, None]
        q_mu = rng.normal(0, 0.3, M)
        L = np.tril(rng.normal(0, 0.2, (M, M)))
        L[np.arange(M), np.arange(M)] = np.abs(L[np.arange(M), np.arange(M)]) + 0.5
        return X, y, Z, q_mu, L

    def _ptgp_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val):
        """Evaluate PTGP whitened-SVGP ELBO at the fixed configuration."""
        ls = pt.scalar("ls")
        eta = pt.scalar("eta")
        q_mu = pt.vector("q_mu")
        q_sqrt = pt.matrix("q_sqrt")
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Poisson(),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=True,
        )
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        elbo_expr = pg.objectives.elbo(svgp, X_var, y_var)
        fn = pytensor.function([X_var, y_var, q_mu, q_sqrt, ls, eta], elbo_expr)
        return float(fn(X, y, q_mu_val, q_sqrt_val, ls_val, eta_val))

    def _gpjax_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val):
        """Evaluate GPJax whitened-SVGP ELBO at the same configuration."""
        kernel = gpx.kernels.Matern52(
            active_dims=[0], lengthscale=jnp.array(ls_val), variance=jnp.array(eta_val**2)
        )
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        likelihood = gpx.likelihoods.Poisson(num_datapoints=X.shape[0])
        posterior = prior * likelihood
        q = gpx.variational_families.WhitenedVariationalGaussian(
            posterior=posterior,
            inducing_inputs=jnp.array(Z),
            variational_mean=jnp.array(q_mu_val)[:, None],
            variational_root_covariance=jnp.array(q_sqrt_val),
            jitter=0.0,
        )
        data = gpx.Dataset(X=jnp.array(X), y=jnp.array(y)[:, None])
        return float(gpx.objectives.elbo(q, data))

    def test_elbo_match(self):
        rng = np.random.default_rng(3)
        X, y, Z, q_mu_val, q_sqrt_val = self._fixed_config(rng)
        ls_val, eta_val = 1.3, 0.9

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val)
        e_gpjax = self._gpjax_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val)

        np.testing.assert_allclose(e_ptgp, e_gpjax, atol=ATOL)