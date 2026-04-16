"""Inference routines for PTGP models.

- ``fit_lbfgsb``: L-BFGS-B optimization for GP and VFE models.
- ``make_training_step``: Returns a JIT-compiled training step for SVGP (user controls the loop).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt

from jax.scipy.optimize import minimize as jax_minimize
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.assumptions.specify import SpecifyAssumptions


# Register JAX dispatch for SpecifyAssumptions (identity op).
@jax_funcify.register(SpecifyAssumptions)
def jax_funcify_SpecifyAssumptions(op, **kwargs):
    def specify_assumptions(x):
        return x

    return specify_assumptions


def _compile_to_jax(inputs, output):
    """Compile a pytensor graph to a JAX function.

    Parameters
    ----------
    inputs : list of TensorVariable
        Symbolic input variables.
    output : TensorVariable
        Symbolic scalar output.

    Returns
    -------
    callable
        JAX function ``(*inputs) -> scalar``.
    """
    f_pt = pytensor.function(inputs, output, mode="JAX")
    fgraph = f_pt.maker.fgraph
    jax_fn_raw = jax_funcify(fgraph)

    def jax_fn(*args):
        return jax_fn_raw(*args)[0]

    return jax_fn


def fit_bfgs(objective_fn, model, X, y, params, init_values, maxiter=1000):
    """Optimize a GP or VFE model using BFGS (via JAX).

    Works in unconstrained space — the user is responsible for
    applying transforms (e.g. ``pt.exp(log_ls)``) when building the model.

    Parameters
    ----------
    objective_fn : callable
        ``(model, X_var, y_var) -> scalar`` returning the quantity to maximize
        (e.g. ``marginal_log_likelihood`` or ``collapsed_elbo``).
    model : GP or VFE
        Model whose kernel/likelihood reference the symbolic parameters.
    X : ndarray, shape (N, D)
        Training inputs.
    y : ndarray, shape (N,)
        Training targets.
    params : list of TensorVariable
        Symbolic variables to optimize. Must be the same variables used
        to build the kernel/likelihood (e.g. ``log_ls``, ``log_sigma``).
    init_values : list of ndarray
        Initial values for each parameter, same order as ``params``.
    maxiter : int
        Maximum number of BFGS iterations.

    Returns
    -------
    opt_values : list of ndarray
        Optimized parameter values, same order as ``params``.
    result : OptimizeResults
        JAX minimize result (has ``.fun``, ``.x``, ``.success``).
    """
    X_var = pt.matrix("_X_opt")
    y_var = pt.vector("_y_opt")

    loss = -objective_fn(model, X_var, y_var)
    jax_loss = _compile_to_jax([*params, X_var, y_var], loss)

    # Pack initial values into a flat vector
    init_flat = [np.asarray(v, dtype=np.float64).ravel() for v in init_values]
    shapes = [np.asarray(v).shape for v in init_values]
    sizes = [v.size for v in init_flat]
    x0 = jnp.concatenate(init_flat) if len(init_flat) > 1 else jnp.array(init_flat[0])

    def _unpack(flat):
        parts = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            part = jnp.reshape(flat[offset : offset + size], shape) if shape else flat[offset]
            offset += size
            parts.append(part)
        return parts

    def packed_loss(flat):
        return jax_loss(*_unpack(flat), X, y)

    result = jax_minimize(packed_loss, x0, method="BFGS", options={"maxiter": maxiter})

    opt_values = [np.asarray(v) for v in _unpack(result.x)]
    return opt_values, result


def fit_model(objective_fn, gp_model, X, y, pm_model=None, maxiter=1000):
    """Optimize a GP or VFE model defined inside a ``pm.Model()`` context.

    Automatically uses PyMC's unconstrained value variables and initial point.

    Parameters
    ----------
    objective_fn : callable
        ``(gp_model, X_var, y_var) -> scalar`` returning the quantity to maximize.
    gp_model : GP or VFE
        PTGP model whose parameters are PyMC RVs.
    X : ndarray, shape (N, D)
        Training inputs.
    y : ndarray, shape (N,)
        Training targets.
    pm_model : pm.Model, optional
        PyMC model context. If None, uses the current model context.
    maxiter : int
        Maximum number of BFGS iterations.

    Returns
    -------
    opt_point : dict
        ``{name: ndarray}`` of optimized unconstrained parameter values.
    result : OptimizeResults
        JAX minimize result.
    """
    import pymc as pm

    pm_model = pm.modelcontext(pm_model)

    X_var = pt.matrix("_X_opt")
    y_var = pt.vector("_y_opt")

    loss = -objective_fn(gp_model, X_var, y_var)
    [loss_replaced] = pm_model.replace_rvs_by_values([loss])

    value_vars = pm_model.continuous_value_vars
    jax_loss = _compile_to_jax([*value_vars, X_var, y_var], loss_replaced)

    ip = pm_model.initial_point()
    var_names = list(ip.keys())
    init_flat = [np.asarray(ip[name], dtype=np.float64).ravel() for name in var_names]
    shapes = [np.asarray(ip[name]).shape for name in var_names]
    sizes = [v.size for v in init_flat]
    x0 = jnp.concatenate(init_flat) if len(init_flat) > 1 else jnp.array(init_flat[0])

    def _unpack(flat):
        parts = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            part = jnp.reshape(flat[offset : offset + size], shape) if shape else flat[offset]
            offset += size
            parts.append(part)
        return parts

    def packed_loss(flat):
        return jax_loss(*_unpack(flat), X, y)

    result = jax_minimize(packed_loss, x0, method="BFGS", options={"maxiter": maxiter})

    opt_values = _unpack(result.x)
    opt_point = {name: np.asarray(v) for name, v in zip(var_names, opt_values)}
    return opt_point, result


def make_training_step(objective_fn, model, X_var, y_var, params, optimizer):
    """Create a JIT-compiled SVGP training step.

    The user controls the training loop and minibatching.

    Parameters
    ----------
    objective_fn : callable
        ``(model, X_var, y_var) -> scalar`` returning the quantity to maximize
        (e.g. ``elbo``).
    model : SVGP
        SVGP model whose kernel/likelihood/q_mu/q_sqrt reference symbolic params.
    X_var : TensorVariable
        Symbolic minibatch input placeholder.
    y_var : TensorVariable
        Symbolic minibatch target placeholder.
    params : list of TensorVariable
        Symbolic variables to optimize (hyperparameters + variational params).
    optimizer : optax optimizer
        Optax optimizer instance (e.g. ``optax.adam(1e-3)``).

    Returns
    -------
    init_fn : callable
        ``(param_values) -> opt_state`` — initialize optimizer state.
    step_fn : callable
        ``(X_batch, y_batch, param_values, opt_state) -> (param_values, opt_state, loss)``
        One training step.  ``param_values`` is a tuple of arrays.
    """
    import optax

    loss = -objective_fn(model, X_var, y_var)
    jax_loss = _compile_to_jax([X_var, y_var, *params], loss)

    def f_loss(X, y, param_tuple):
        return jax_loss(X, y, *param_tuple)

    def init_fn(param_values):
        return optimizer.init(param_values)

    @jax.jit
    def step_fn(X_batch, y_batch, param_values, opt_state):
        loss_val, grads = jax.value_and_grad(f_loss, argnums=2)(X_batch, y_batch, param_values)
        updates, opt_state = optimizer.update(grads, opt_state, param_values)
        param_values = optax.apply_updates(param_values, updates)
        return param_values, opt_state, loss_val

    return init_fn, step_fn


def make_training_step_model(
    objective_fn, gp_model, X_var, y_var, extra_params=None, optimizer=None, pm_model=None
):
    """Create a JIT-compiled SVGP training step using a ``pm.Model()`` context.

    Automatically uses PyMC's unconstrained value variables. Additional
    parameters (e.g. variational ``q_mu``, ``q_sqrt``) that are not PyMC RVs
    can be passed via ``extra_params``.

    Parameters
    ----------
    objective_fn : callable
        ``(gp_model, X_var, y_var) -> scalar`` returning the quantity to maximize.
    gp_model : SVGP
        PTGP model whose hyperparameters are PyMC RVs.
    X_var : TensorVariable
        Symbolic minibatch input placeholder.
    y_var : TensorVariable
        Symbolic minibatch target placeholder.
    extra_params : list of TensorVariable, optional
        Non-PyMC symbolic variables to optimize (e.g. ``q_mu``, ``q_sqrt``).
    optimizer : optax optimizer
        Optax optimizer instance.
    pm_model : pm.Model, optional
        PyMC model context. If None, uses the current model context.

    Returns
    -------
    init_fn : callable
        ``(param_values) -> opt_state``
    step_fn : callable
        ``(X_batch, y_batch, param_values, opt_state) -> (param_values, opt_state, loss)``
        ``param_values`` is a tuple: ``(*value_var_values, *extra_param_values)``.
    """
    import optax
    import pymc as pm

    pm_model = pm.modelcontext(pm_model)
    if extra_params is None:
        extra_params = []

    loss = -objective_fn(gp_model, X_var, y_var)
    [loss_replaced] = pm_model.replace_rvs_by_values([loss])

    value_vars = pm_model.continuous_value_vars
    all_params = [*value_vars, *extra_params]
    jax_loss = _compile_to_jax([X_var, y_var, *all_params], loss_replaced)

    def f_loss(X, y, param_tuple):
        return jax_loss(X, y, *param_tuple)

    def init_fn(param_values):
        return optimizer.init(param_values)

    @jax.jit
    def step_fn(X_batch, y_batch, param_values, opt_state):
        loss_val, grads = jax.value_and_grad(f_loss, argnums=2)(X_batch, y_batch, param_values)
        updates, opt_state = optimizer.update(grads, opt_state, param_values)
        param_values = optax.apply_updates(param_values, updates)
        return param_values, opt_state, loss_val

    return init_fn, step_fn
