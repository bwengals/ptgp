"""Likelihood tests against GPJax reference and analytical results."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from gpjax.likelihoods import Gaussian as GPJaxGaussian, Bernoulli as GPJaxBernoulli
from gpjax.integrators import AnalyticalGaussianIntegrator, GHQuadratureIntegrator

from ptgp.likelihoods import Gaussian, Bernoulli, StudentT, Poisson, NegativeBinomial

ATOL = 1e-5


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestGaussian:
    def test_ve_against_gpjax(self):
        mu, var = np.array([0.0, 0.5, -1.0]), np.array([0.1, 0.5, 1.0])
        y, sigma = np.array([0.1, 0.3, -0.8]), 0.5

        ve = _eval(Gaussian(sigma).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        gpjax_ve = np.array(GPJaxGaussian(
            num_datapoints=3, obs_stddev=jnp.array(sigma),
            integrator=AnalyticalGaussianIntegrator(),
        ).expected_log_likelihood(
            y=jnp.array(y)[:, None], mean=jnp.array(mu)[:, None], variance=jnp.array(var)[:, None]))

        np.testing.assert_allclose(ve, gpjax_ve, atol=ATOL)

    def test_zero_var_matches_log_prob(self):
        mu, y, sigma = np.array([0.0, 1.0]), np.array([0.1, 0.9]), 0.3
        lik = Gaussian(sigma)
        ve = _eval(lik.variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(np.zeros(2))))
        lp = _eval(lik._log_prob(pt.as_tensor_variable(mu), pt.as_tensor_variable(y)))
        np.testing.assert_allclose(ve, lp, atol=1e-12)

    def test_predict_mean_and_var(self):
        mu, var, sigma = np.array([1.0, 2.0]), np.array([0.5, 1.0]), 0.3
        pm, pv = _eval(*Gaussian(sigma).predict_mean_and_var(
            pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        np.testing.assert_allclose(pm, mu, atol=1e-12)
        np.testing.assert_allclose(pv, var + sigma**2, atol=1e-12)

    def test_predict_log_density(self):
        mu, var = np.array([0.0, 1.0]), np.array([0.1, 0.5])
        y, sigma = np.array([0.1, 0.8]), 0.5
        pld = _eval(Gaussian(sigma).predict_log_density(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        # Manual: log N(y; mu, var + sigma^2)
        total_var = var + sigma**2
        expected = -0.5 * (np.log(2 * np.pi * total_var) + (y - mu)**2 / total_var)
        np.testing.assert_allclose(pld, expected, atol=1e-12)


class TestBernoulli:
    def test_ve_against_gpjax(self):
        mu, var = np.array([0.0, 1.0, -1.0]), np.array([0.25, 0.5, 1.0])
        y = np.array([1.0, 1.0, 0.0])

        ve = _eval(Bernoulli(n_points=20).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        gpjax_ve = np.array(GPJaxBernoulli(
            num_datapoints=3, integrator=GHQuadratureIntegrator(num_points=20),
        ).expected_log_likelihood(
            y=jnp.array(y)[:, None], mean=jnp.array(mu)[:, None], variance=jnp.array(var)[:, None]))

        np.testing.assert_allclose(ve, gpjax_ve, atol=ATOL)

    def test_predict_mean_and_var_closed_form(self):
        mu, var = np.array([0.0, 2.0, -2.0]), np.array([0.1, 0.5, 1.0])
        pm_val, pv_val = _eval(*Bernoulli().predict_mean_and_var(
            pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        assert np.all(pm_val >= 0.0) and np.all(pm_val <= 1.0)
        assert np.all(pv_val >= 0.0) and np.all(pv_val <= 0.25)

    def test_ve_negative(self):
        mu, var = np.array([0.0, 2.0, -2.0]), np.array([0.1, 0.5, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        ve = _eval(Bernoulli().variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        assert np.all(ve <= 0.0)


class TestStudentT:
    def test_converges_to_gaussian(self):
        mu, var = np.array([0.0, 0.5]), np.array([0.1, 0.3])
        y, sigma = np.array([0.1, 0.4]), 0.5

        ve_gauss = _eval(Gaussian(sigma).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        ve_student = _eval(StudentT(nu=1000.0, sigma=sigma).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        np.testing.assert_allclose(ve_student, ve_gauss, atol=1e-2)

    def test_quadrature_convergence(self):
        mu, var, y = np.array([0.0]), np.array([1.0]), np.array([0.5])
        ve_20 = _eval(StudentT(nu=5.0, sigma=1.0, n_points=20).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        ve_50 = _eval(StudentT(nu=5.0, sigma=1.0, n_points=50).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        np.testing.assert_allclose(ve_20, ve_50, atol=1e-6)


class TestPoisson:
    def test_closed_form_matches_quadrature(self):
        """Poisson with log link has a closed-form VE — should match quadrature."""
        mu, var = np.array([0.0, 1.0, -0.5]), np.array([0.1, 0.5, 1.0])
        y = np.array([1.0, 3.0, 0.0])

        ve_closed = _eval(Poisson().variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        # Use base class quadrature via _gauss_hermite directly
        lik = Poisson(n_points=50)
        ve_quad = _eval(lik._gauss_hermite(lik._log_prob, pt.as_tensor_variable(y),
            pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        np.testing.assert_allclose(ve_closed, ve_quad, atol=1e-6)

    def test_ve_values(self):
        """Spot check: with mu=0, var=0, y=1, VE = 1*0 - exp(0) - log(1!) = -1."""
        ve = _eval(Poisson().variational_expectation(
            pt.as_tensor_variable(np.array([1.0])),
            pt.as_tensor_variable(np.array([0.0])),
            pt.as_tensor_variable(np.array([0.0]))))
        np.testing.assert_allclose(ve, np.array([-1.0]), atol=1e-12)


class TestNegativeBinomial:
    def test_quadrature_convergence(self):
        mu, var = np.array([0.5, 1.0]), np.array([0.2, 0.5])
        y = np.array([2.0, 5.0])

        ve_20 = _eval(NegativeBinomial(alpha=5.0, n_points=20).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        ve_50 = _eval(NegativeBinomial(alpha=5.0, n_points=50).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        np.testing.assert_allclose(ve_20, ve_50, atol=1e-6)

    def test_converges_to_poisson(self):
        """NB with large alpha should approach Poisson."""
        mu, var = np.array([0.5, 1.0]), np.array([0.1, 0.3])
        y = np.array([1.0, 3.0])

        # Use quadrature for both so comparison is apples-to-apples
        poisson_lik = Poisson(n_points=50)
        ve_poisson = _eval(poisson_lik._gauss_hermite(poisson_lik._log_prob,
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        ve_nb = _eval(NegativeBinomial(alpha=1e4, n_points=50).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        np.testing.assert_allclose(ve_nb, ve_poisson, atol=1e-2)


class TestConfigurableLink:
    def test_bernoulli_logit_link(self):
        """Bernoulli with logit link should differ from probit but still be valid."""
        mu, var = np.array([0.0, 1.0]), np.array([0.5, 0.5])
        y = np.array([1.0, 0.0])

        ve_probit = _eval(Bernoulli().variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        ve_logit = _eval(Bernoulli(invlink=pt.sigmoid).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))

        assert np.all(ve_probit < 0) and np.all(ve_logit < 0)
        assert not np.allclose(ve_probit, ve_logit)

    def test_poisson_custom_link_uses_quadrature(self):
        """Poisson with non-exp link should fall back to quadrature and still work."""
        mu, var = np.array([1.0]), np.array([0.1])
        y = np.array([2.0])

        softplus = lambda f: pt.log1p(pt.exp(f))
        ve = _eval(Poisson(invlink=softplus).variational_expectation(
            pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)))
        assert np.isfinite(ve).all()
