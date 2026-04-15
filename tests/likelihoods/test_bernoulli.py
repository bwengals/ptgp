"""Bernoulli likelihood tests against GPJax reference and analytical results."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt

from gpjax.likelihoods import Bernoulli as GPJaxBernoulli
from gpjax.integrators import GHQuadratureIntegrator

from ptgp.likelihoods import Bernoulli

ATOL = 1e-5


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


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
