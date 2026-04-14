"""Optimizers for PyTensor shared variables.

Each optimizer takes a loss expression and a list of shared variables,
and returns an OrderedDict mapping each shared variable to its update
expression. Pass this dict as ``updates`` to ``pytensor.function``.

Ported from ``pymc.variational.updates`` (originally from Lasagne).
"""

from collections import OrderedDict

import numpy as np
import pytensor
import pytensor.tensor as pt


def _get_grads(loss, params):
    for p in params:
        if not isinstance(p, pytensor.compile.SharedVariable):
            raise ValueError("params must contain shared variables only")
    return pytensor.grad(loss, params)


def sgd(loss, params, learning_rate=1e-3):
    """Stochastic gradient descent."""
    grads = _get_grads(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def adam(loss, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam optimizer (Kingma & Ba, 2014)."""
    grads = _get_grads(loss, params)
    t_prev = pytensor.shared(np.float64(0.0))
    updates = OrderedDict()

    one = pt.constant(1)
    t = t_prev + 1
    a_t = learning_rate * pt.sqrt(one - beta2**t) / (one - beta1**t)

    for param, g_t in zip(params, grads):
        value = param.get_value(borrow=True)
        m_prev = pytensor.shared(
            np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape,
        )
        v_prev = pytensor.shared(
            np.zeros(value.shape, dtype=value.dtype), shape=param.type.shape,
        )

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * g_t**2
        step = a_t * m_t / (pt.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates
