"""Training and prediction compilation for PTGP models with PyMC priors.

Uses ``pytensor.shared`` variables so that training automatically updates
the parameters used by prediction — no model reconstruction needed.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph.replace import graph_replace

from ptgp.optim.optimizers import adam


def _make_shared_params(pm_model, extra_vars=None, extra_init=None):
    """Create shared variables for a PyMC model's value vars and any extras.

    Returns
    -------
    shared_params : dict
        ``{value_var: shared_var}`` for PyMC value vars.
    shared_extras : list
        Shared variables for extra (non-PyMC) parameters.
    all_shared : list
        All shared variables in order: value vars then extras.
    """
    ip = pm_model.initial_point()
    shared_params = {}
    for vv in pm_model.continuous_value_vars:
        shared_params[vv] = pytensor.shared(
            np.asarray(ip[vv.name], dtype=np.float64),
            name=vv.name,
        )

    shared_extras = []
    if extra_vars is not None:
        for var, init in zip(extra_vars, extra_init):
            shared_extras.append(pytensor.shared(np.asarray(init, dtype=np.float64), name=var.name))

    all_shared = list(shared_params.values()) + shared_extras
    return shared_params, shared_extras, all_shared


def _replace_graph(
    outputs,
    pm_model,
    shared_params,
    extra_vars=None,
    shared_extras=None,
    frozen_vars=None,
):
    """Replace PyMC RVs with value vars, then value vars with shared vars.

    Also replaces any extra symbolic vars with their shared counterparts,
    and any ``frozen_vars`` with their (non-trainable) constant tensors.
    Only includes replacements for variables that actually appear in the
    graph, since not all parameters may be used (e.g. likelihood sigma is
    absent from SVGP's predict graph).
    """
    replaced = pm_model.replace_rvs_by_values(outputs)
    replace_map = dict(shared_params)
    if extra_vars is not None and shared_extras is not None:
        for var, sv in zip(extra_vars, shared_extras):
            replace_map[var] = sv
    if frozen_vars is not None:
        for var, value in frozen_vars.items():
            replace_map[var] = pt.as_tensor_variable(np.asarray(value, dtype=np.float64))

    # strict=False because not all parameters may appear in every graph
    # (e.g. likelihood sigma is absent from SVGP's predict graph)
    return [graph_replace(r, replace_map, strict=False) for r in replaced]


def compile_training_step(
    objective_fn,
    gp_model,
    X_var,
    y_var,
    pm_model=None,
    optimizer_fn=None,
    extra_vars=None,
    extra_init=None,
    frozen_vars=None,
    param_groups=None,
    **optimizer_kwargs,
):
    """Compile a training step function for a PTGP model with PyMC priors.

    Parameters
    ----------
    objective_fn : callable
        ``(gp_model, X_var, y_var) -> scalar`` returning the quantity to
        maximize (e.g. ``elbo``, ``marginal_log_likelihood``).
    gp_model : GP, VFE, or SVGP
        PTGP model whose hyperparameters are PyMC RVs.
    X_var : TensorVariable
        Symbolic input placeholder.
    y_var : TensorVariable
        Symbolic target placeholder.
    pm_model : pm.Model, optional
        PyMC model context. Uses current context if None. Every continuous
        free RV in the model is automatically made into a trainable shared
        variable — you do not need to list them.
    optimizer_fn : callable, optional
        Optimizer function (default: ``adam``). Must have signature
        ``(loss, params, **kwargs) -> updates_dict``.
    extra_vars : list of TensorVariable, optional
        Additional symbolic variables to optimize that are not PyMC RVs and
        so cannot be discovered from ``pm_model``. Typical entries: SVGP
        ``q_mu`` / ``q_sqrt``, or a trainable inducing-point ``Z_var``.
    extra_init : list of ndarray, optional
        Initial values for ``extra_vars``, in the same order. Required
        whenever ``extra_vars`` is provided.
    frozen_vars : dict[TensorVariable, ndarray], optional
        Symbolic variables to pin to constant values for this compile call.
        Each key is replaced in the graph by ``pt.as_tensor_variable(value)``
        before compilation, so it receives no gradient and is not updated.
        Use for staged training where a single model is built once with
        symbolic placeholders: in an early phase, pass the placeholder in
        ``frozen_vars`` (and omit it from ``extra_vars``) to freeze it; in
        a later phase, move it to ``extra_vars`` to make it trainable.
        Keys must not also appear in ``extra_vars``.
    param_groups : dict[str, list[TensorVariable]], optional
        Maps a group name to a list of symbolic variables (PyMC value vars
        or entries of ``extra_vars``). Resolved to shared variables and
        forwarded to the optimizer. Required when ``learning_rate`` is a
        dict. The union of groups must cover every optimized parameter.
    **optimizer_kwargs
        Passed to the optimizer (e.g. ``learning_rate=1e-2`` or a dict
        of per-group rates).

    Returns
    -------
    train_step : callable
        ``(X_batch, y_batch) -> loss_value``. Updates shared parameters
        in place.
    shared_params : dict
        ``{value_var: shared_var}`` — the shared variables holding the
        unconstrained parameter values. Needed by ``compile_predict``.
    shared_extras : list
        Shared variables for ``extra_vars``. Needed by ``compile_predict``.
    """
    pm_model = pm.modelcontext(pm_model)
    if optimizer_fn is None:
        optimizer_fn = adam

    if frozen_vars and extra_vars:
        overlap = [v for v in extra_vars if v in frozen_vars]
        if overlap:
            names = [v.name or repr(v) for v in overlap]
            raise ValueError(
                f"Variables appear in both extra_vars and frozen_vars: {names}. "
                f"They cannot be both trainable and frozen."
            )

    shared_params, shared_extras, all_shared = _make_shared_params(
        pm_model,
        extra_vars,
        extra_init,
    )

    if param_groups is not None:
        sym_to_shared = dict(shared_params)
        if extra_vars is not None:
            for var, sv in zip(extra_vars, shared_extras):
                sym_to_shared[var] = sv
        resolved_groups = {}
        for name, group in param_groups.items():
            resolved = []
            for var in group:
                if var not in sym_to_shared:
                    raise ValueError(
                        f"param_groups[{name!r}] contains unknown variable "
                        f"{var.name or repr(var)}"
                    )
                resolved.append(sym_to_shared[var])
            resolved_groups[name] = resolved
        optimizer_kwargs = {**optimizer_kwargs, "param_groups": resolved_groups}

    loss = -objective_fn(gp_model, X_var, y_var)
    [loss_replaced] = _replace_graph(
        [loss],
        pm_model,
        shared_params,
        extra_vars,
        shared_extras,
        frozen_vars=frozen_vars,
    )

    updates = optimizer_fn(loss_replaced, all_shared, **optimizer_kwargs)

    train_step = pytensor.function(
        [X_var, y_var],
        loss_replaced,
        updates=updates,
    )
    return train_step, shared_params, shared_extras


def get_trained_params(pm_model, shared_params):
    """Get trained hyperparameter values in the original (constrained) space.

    Parameters
    ----------
    pm_model : pm.Model
        The PyMC model context used in training.
    shared_params : dict
        ``{value_var: shared_var}`` from ``compile_training_step``.

    Returns
    -------
    dict
        ``{rv_name: constrained_value}`` for each free RV in the model.
    """
    result = {}
    for rv in pm_model.free_RVs:
        vv = pm_model.rvs_to_values[rv]
        transform = pm_model.rvs_to_transforms[rv]
        unconstrained = shared_params[vv].get_value()
        constrained = transform.backward(unconstrained).eval()
        result[rv.name] = float(constrained) if constrained.ndim == 0 else constrained
    return result


def compile_predict(
    gp_model,
    X_new_var,
    pm_model,
    shared_params,
    extra_vars=None,
    shared_extras=None,
    X_train=None,
    y_train=None,
    incl_lik=False,
):
    """Compile a prediction function that reads trained shared parameters.

    Parameters
    ----------
    gp_model : GP, VFE, or SVGP
        The same PTGP model object used in training.
    X_new_var : TensorVariable
        Symbolic variable for prediction inputs.
    pm_model : pm.Model
        The PyMC model context used in training.
    shared_params : dict
        ``{value_var: shared_var}`` from ``compile_training_step``.
    extra_vars : list of TensorVariable, optional
        Non-PyMC symbolic variables (same as used in training).
    shared_extras : list, optional
        Shared variables for ``extra_vars`` (from ``compile_training_step``).
    X_train : ndarray, optional
        Training inputs (required for GP and VFE).
    y_train : ndarray, optional
        Training targets (required for GP and VFE).
    incl_lik : bool
        If True, include likelihood noise in the predictions.

    Returns
    -------
    predict_fn : callable
        ``(X_new) -> (mean, var)`` using the trained parameter values.
    """
    if X_train is not None:
        mean, var = gp_model.predict(
            X_new_var,
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
            incl_lik=incl_lik,
        )
    else:
        mean, var = gp_model.predict(X_new_var, incl_lik=incl_lik)

    [mean_s, var_s] = _replace_graph(
        [mean, var],
        pm_model,
        shared_params,
        extra_vars,
        shared_extras,
    )

    return pytensor.function([X_new_var], [mean_s, var_s])
