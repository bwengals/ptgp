"""Shared fixtures and helpers for PTGP tests.

GPJax is used as the reference implementation for numerical correctness.
The helper ``make_params`` converts a flat dict of numeric values into
the parameter formats expected by both PTGP and GPJax so that all tests
can share a single source of truth for hyperparameter values.
"""

import jax.numpy as jnp
import pytest


@pytest.fixture
def X_1d():
    """20 evenly spaced points in [0, 5], shape (20, 1)."""
    return jnp.linspace(0.0, 5.0, 20)[:, None]


@pytest.fixture
def X_2d():
    """40 points on a 2D grid, shape (40, 2)."""
    x = jnp.linspace(0.0, 1.0, 20)
    return jnp.stack([x, x[::-1]], axis=1)


@pytest.fixture
def rng_key():
    import jax

    return jax.random.PRNGKey(0)
