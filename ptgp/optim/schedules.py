"""Learning rate schedules for PTGP optimizers.

Each builder returns a callable ``schedule(t) -> lr``, where ``t`` is a
symbolic iteration counter (starting at 1) and ``lr`` is a symbolic scalar.
Optimizers in :mod:`ptgp.optim.optimizers` accept either a plain scalar,
a builder from this module, or a dict mapping group name to either.
"""

import numpy as np
import pytensor.tensor as pt


def constant(lr):
    """Constant learning rate.

    Parameters
    ----------
    lr : float
        Fixed learning rate.

    Returns
    -------
    callable
        ``schedule(t) -> lr`` that ignores ``t``.
    """

    def schedule(t):
        return pt.constant(lr, dtype="float64")

    return schedule


def exponential_decay(base, decay_rate=0.9, decay_steps=1000):
    """Geometric decay: ``lr(t) = base * decay_rate ** (t / decay_steps)``.

    Monotone, budget-agnostic. Every ``decay_steps`` iterations the LR is
    multiplied by ``decay_rate``. Asymptotes toward zero.

    Parameters
    ----------
    base : float
        Initial learning rate at ``t=0``.
    decay_rate : float, optional
        Multiplicative factor per ``decay_steps`` iterations. Default 0.9.
    decay_steps : float, optional
        Iterations per decay. Default 1000.
    """

    def schedule(t):
        return base * decay_rate ** (t / decay_steps)

    return schedule


def cosine(base, T_max, lr_min=0.0):
    """Cosine annealing: smooth decay from ``base`` to ``lr_min`` over ``T_max`` steps.

    ``lr(t) = lr_min + 0.5 * (base - lr_min) * (1 + cos(pi * min(t, T_max) / T_max))``.
    After ``T_max`` the LR stays at ``lr_min`` (no warm restarts).

    Parameters
    ----------
    base : float
        Initial learning rate at ``t=0``.
    T_max : float
        Number of iterations over which to decay from ``base`` to ``lr_min``.
    lr_min : float, optional
        Final learning rate. Default 0.0.
    """

    def schedule(t):
        t_clipped = pt.minimum(t, T_max)
        return lr_min + 0.5 * (base - lr_min) * (1.0 + pt.cos(np.pi * t_clipped / T_max))

    return schedule
