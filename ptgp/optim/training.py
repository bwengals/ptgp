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


def _make_shared_params(model, extra_vars=None, extra_init=None):
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
    ip = model.initial_point()
    shared_params = {}
    for vv in model.continuous_value_vars:
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
    model,
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
    replaced = model.replace_rvs_by_values(outputs)
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
    model=None,
    optimizer_fn=None,
    extra_vars=None,
    extra_init=None,
    frozen_vars=None,
    param_groups=None,
    include_prior=True,
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
    model : pm.Model, optional
        PyMC model. Uses the enclosing ``with pm.Model()`` context if
        None. Every continuous free RV in the model is automatically
        made into a trainable shared variable — you do not need to list
        them.
    optimizer_fn : callable, optional
        Optimizer function (default: ``adam``). Must have signature
        ``(loss, params, **kwargs) -> updates_dict``.
    include_prior : bool
        If True (default), add the PyMC joint log-prior (with the
        transform log-det-jacobian) to the objective, yielding MAP in
        the unconstrained space. Set False for MLE / pure ELBO without
        prior regularization.
    extra_vars : list of TensorVariable, optional
        Additional symbolic variables to optimize that are not PyMC RVs and
        so cannot be discovered from ``model``. Typical entries: SVGP
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
    model = pm.modelcontext(model)
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
        model,
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
    if include_prior:
        loss = loss - model.logp(jacobian=True, sum=True)
    [loss_replaced] = _replace_graph(
        [loss],
        model,
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


def compile_scipy_objective(
    objective_fn,
    gp_model,
    X_var,
    y_var,
    model=None,
    extra_vars=None,
    extra_init=None,
    frozen_vars=None,
    include_prior=True,
):
    """Compile a (loss, grad) objective for ``scipy.optimize.minimize``.

    Unlike :func:`compile_training_step`, scipy owns the parameter vector
    during optimization: the compiled function takes a flat ``theta`` as
    input, returns the scalar loss and flat gradient, and does not mutate
    any state. Once scipy converges, call ``unpack_to_shared(result.x)``
    to write the final values into the shared variables used by
    :func:`compile_predict`.

    Parameters
    ----------
    objective_fn : callable
        ``(gp_model, X_var, y_var) -> scalar``, returning the quantity to
        maximize (e.g. ``marginal_log_likelihood``, ``collapsed_elbo``).
        The returned scalar is negated internally so scipy minimizes.
    gp_model : GP, VFE, or SVGP
        PTGP model whose hyperparameters are PyMC RVs.
    X_var : TensorVariable
        Symbolic input placeholder. ``X`` is passed to the compiled
        function on each scipy iteration — typically the full training
        inputs for GP/VFE (batching is not used with quasi-Newton methods).
    y_var : TensorVariable
        Symbolic target placeholder, handled like ``X_var``.
    model : pm.Model, optional
        PyMC model. Uses the enclosing ``with pm.Model()`` context if
        None. Every continuous free RV becomes a slice of ``theta``; you
        do not need to list them.
    extra_vars : list of TensorVariable, optional
        Additional symbolic variables to optimize that are not PyMC RVs
        (so cannot be discovered from ``model``). Typical entries: VFE
        inducing-point ``Z_var``, or SVGP ``q_mu`` / ``q_sqrt``.
    extra_init : list of ndarray, optional
        Initial values for ``extra_vars``, in the same order. Required
        whenever ``extra_vars`` is provided. Shapes determine the
        corresponding segments of ``theta``.
    frozen_vars : dict[TensorVariable, ndarray], optional
        Symbolic variables to pin to constant values. Each key is
        replaced in the graph by ``pt.as_tensor_variable(value)`` before
        compilation, receives no gradient, and is excluded from
        ``theta``. Keys must not also appear in ``extra_vars``.
    include_prior : bool
        If True (default), add the PyMC joint log-prior (with the
        transform log-det-jacobian) to the objective, yielding MAP in
        the unconstrained space. Set False for MLE / pure ELBO without
        prior regularization.

    Returns
    -------
    fun : callable
        ``(theta, X, y) -> (loss, flat_grad)``. ``theta`` is a 1D
        ``ndarray``; ``loss`` is a scalar; ``flat_grad`` is a 1D
        ``ndarray`` with the same layout as ``theta``. Pass directly to
        ``scipy.optimize.minimize(fun, theta0, args=(X, y), jac=True,
        method=...)``.
    theta0 : ndarray
        Flat initial parameter vector. Layout: PyMC value vars in
        ``model.continuous_value_vars`` order, followed by
        ``extra_vars`` in the order given. Values come from
        ``model.initial_point()`` and ``extra_init``. Use as the
        ``x0`` argument to ``scipy.optimize.minimize``.
    unpack_to_shared : callable
        ``(theta) -> None``. Slices ``theta`` along the same layout as
        ``theta0`` and writes each piece into the corresponding entry of
        ``shared_params`` / ``shared_extras``. Call once after scipy
        converges so that :func:`compile_predict` sees the trained
        values. Mutates the captured shared vars in place; returns
        nothing.
    shared_params : dict
        ``{value_var: shared_var}`` for every continuous PyMC value var.
        Needed by :func:`compile_predict` and :func:`get_trained_params`.
        Not read by ``fun`` — present only for the predict handoff.
    shared_extras : list
        Shared variables for ``extra_vars``, in the same order. Needed
        by :func:`compile_predict`. Not read by ``fun``.
    """
    model = pm.modelcontext(model)

    shared_params, shared_extras, _ = _make_shared_params(
        model, extra_vars, extra_init,
    )

    value_vars_ordered = list(model.continuous_value_vars)
    layout = []
    theta0_pieces = []
    for vv in value_vars_ordered:
        sv = shared_params[vv]
        val = sv.get_value()
        layout.append((sv, val.shape, val.size))
        theta0_pieces.append(val.ravel())
    if extra_vars is not None:
        for sv in shared_extras:
            val = sv.get_value()
            layout.append((sv, val.shape, val.size))
            theta0_pieces.append(val.ravel())
    theta0 = np.concatenate(theta0_pieces) if theta0_pieces else np.zeros(0)

    theta_var = pt.vector("_theta", dtype="float64")
    pieces = []
    offset = 0
    for _, shape, size in layout:
        pieces.append(theta_var[offset:offset + size].reshape(shape))
        offset += size

    loss = -objective_fn(gp_model, X_var, y_var)
    if include_prior:
        loss = loss - model.logp(jacobian=True, sum=True)
    [loss_rvs_replaced] = model.replace_rvs_by_values([loss])

    replace_map = {}
    piece_iter = iter(pieces)
    for vv in value_vars_ordered:
        replace_map[vv] = next(piece_iter)
    if extra_vars is not None:
        for var in extra_vars:
            replace_map[var] = next(piece_iter)
    if frozen_vars is not None:
        for var, value in frozen_vars.items():
            replace_map[var] = pt.as_tensor_variable(np.asarray(value, dtype=np.float64))

    loss_replaced = graph_replace(loss_rvs_replaced, replace_map, strict=False)
    flat_grad = pt.grad(loss_replaced, theta_var)

    fun = pytensor.function([theta_var, X_var, y_var], [loss_replaced, flat_grad])

    def unpack_to_shared(theta):
        """Write ``theta`` into the captured shared vars for prediction."""
        theta = np.asarray(theta, dtype=np.float64)
        offset = 0
        for sv, shape, size in layout:
            sv.set_value(theta[offset:offset + size].reshape(shape))
            offset += size

    return fun, theta0, unpack_to_shared, shared_params, shared_extras


def get_trained_params(model, shared_params):
    """Get trained hyperparameter values in the original (constrained) space.

    Parameters
    ----------
    model : pm.Model
        The PyMC model used in training.
    shared_params : dict
        ``{value_var: shared_var}`` from ``compile_training_step``.

    Returns
    -------
    dict
        ``{rv_name: constrained_value}`` for each free RV in the model.
    """
    result = {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms[rv]
        unconstrained = shared_params[vv].get_value()
        if transform is None:
            constrained = np.asarray(unconstrained)
        else:
            constrained = transform.backward(unconstrained).eval()
        result[rv.name] = float(constrained) if constrained.ndim == 0 else constrained
    return result


def compile_predict(
    gp_model,
    X_new_var,
    model,
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
    model : pm.Model
        The PyMC model used in training.
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
        mean, var = gp_model.predict_marginal(
            X_new_var,
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
            incl_lik=incl_lik,
        )
    else:
        mean, var = gp_model.predict_marginal(X_new_var, incl_lik=incl_lik)

    [mean_s, var_s] = _replace_graph(
        [mean, var],
        model,
        shared_params,
        extra_vars,
        shared_extras,
    )

    return pytensor.function([X_new_var], [mean_s, var_s])