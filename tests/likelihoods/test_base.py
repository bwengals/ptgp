"""Tests for the likelihood base class — configurable inverse-link behavior."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.likelihoods import Bernoulli, Poisson


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


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
