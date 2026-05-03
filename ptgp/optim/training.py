"""Training and prediction compilation for PTGP models with PyMC priors.

Uses ``pytensor.shared`` variables so that training automatically updates
the parameters used by prediction — no model reconstruction needed.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph.replace import graph_replace

from ptgp.objectives import vfe_diagnostics
from ptgp.optim.optimizers import adam


def _make_initial_point(model, init="default", rng=None, n_median_samples=500):
    """Return an initial-point dict (unconstrained space) using ``init`` strategy.

    Parameters
    ----------
    model : pm.Model
    init : str
        ``"default"``      -- PyMC's ``model.initial_point()``; 0 in
                              unconstrained space for every parameter unless
                              ``initval`` was set explicitly.
        ``"prior_draw"``   -- one draw from each prior distribution,
                              transformed to unconstrained space.
        ``"prior_median"`` -- median of each prior (estimated from
                              ``n_median_samples`` draws), transformed to
                              unconstrained space.
    rng : int or numpy Generator, optional
    n_median_samples : int
        Number of prior samples used to estimate the median.  Only used
        when ``init="prior_median"``.

    Returns
    -------
    dict
        ``{value_var_name: unconstrained_array}``, same layout as
        ``model.initial_point()``.
    """
    if init == "default":
        return model.initial_point()

    rng = np.random.default_rng(rng)
    ip = {}

    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        seed = int(rng.integers(0, 2**31))

        if init == "prior_draw":
            val = np.asarray(pm.draw(rv, random_seed=seed), dtype=np.float64)
        elif init == "prior_median":
            samples = pm.draw(rv, draws=n_median_samples, random_seed=seed)
            val = np.asarray(np.median(samples, axis=0), dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown init strategy {init!r}. "
                "Expected 'default', 'prior_draw', or 'prior_median'."
            )

        transform = model.rvs_to_transforms.get(rv)
        if transform is not None:
            unconstrained = np.asarray(
                transform.forward(pt.as_tensor_variable(val)).eval(),
                dtype=np.float64,
            )
        else:
            unconstrained = val

        ip[vv.name] = unconstrained

    return ip


def _make_shared_params(model, extra_vars=None, extra_init=None,
                        init="default", init_rng=None, frozen_vars=None):
    """Create shared variables for a PyMC model's value vars and any extras.

    If ``frozen_vars`` is given, value vars appearing as keys are initialized
    to their freeze values rather than the PyMC initial point. Without this,
    a frozen var's shared slot (and its scipy theta slice) would carry the
    initial point — wrong for diagnostics and for ``unpack_to_shared``.

    Returns
    -------
    shared_params : dict
        ``{value_var: shared_var}`` for PyMC value vars.
    shared_extras : list
        Shared variables for extra (non-PyMC) parameters.
    all_shared : list
        All shared variables in order: value vars then extras.
    """
    ip = _make_initial_point(model, init=init, rng=init_rng)
    shared_params = {}
    for vv in model.continuous_value_vars:
        if frozen_vars is not None and vv in frozen_vars:
            init_val = np.asarray(frozen_vars[vv], dtype=np.float64)
        else:
            init_val = np.asarray(ip[vv.name], dtype=np.float64)
        shared_params[vv] = pytensor.shared(init_val, name=vv.name)

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
    compile_kwargs=None,
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
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function``. Use this
        to set ``mode`` (e.g. ``"NUMBA"``, ``"JAX"``), ``allow_input_downcast``,
        etc. Same pattern as ``pm.sample``'s ``compile_kwargs``.
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
        frozen_vars=frozen_vars,
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

    compile_kwargs = dict(compile_kwargs) if compile_kwargs else {}
    extra_updates = compile_kwargs.pop("updates", {})
    train_step = pytensor.function(
        [X_var, y_var],
        loss_replaced,
        updates={**updates, **extra_updates},
        **compile_kwargs,
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
    compile_kwargs=None,
    init="default",
    init_rng=None,
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
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function``. Use this
        to set ``mode`` (e.g. ``"NUMBA"``, ``"JAX"``), ``allow_input_downcast``,
        etc. Same pattern as ``pm.sample``'s ``compile_kwargs``.
    init : str
        Strategy for setting ``theta0``.  One of:

        ``"default"``      PyMC ``model.initial_point()``: 0 in
                           unconstrained space for every parameter unless
                           ``initval`` was set explicitly. Equivalent to
                           ``exp(0) = 1`` for all positive parameters.
        ``"prior_draw"``   Draw once from each prior and transform to
                           unconstrained space.  Adds stochasticity; use
                           ``init_rng`` for reproducibility.
        ``"prior_median"`` Median of each prior (estimated from 500 draws)
                           transformed to unconstrained space.  Deterministic
                           given ``init_rng``; more central than a single draw.
    init_rng : int or numpy Generator, optional
        Seed for ``"prior_draw"`` and ``"prior_median"`` strategies.

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
        model, extra_vars, extra_init, init=init, init_rng=init_rng,
        frozen_vars=frozen_vars,
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

    fun = pytensor.function(
        [theta_var, X_var, y_var],
        [loss_replaced, flat_grad],
        **(compile_kwargs or {}),
    )

    def unpack_to_shared(theta):
        """Write ``theta`` into the captured shared vars for prediction."""
        theta = np.asarray(theta, dtype=np.float64)
        offset = 0
        for sv, shape, size in layout:
            sv.set_value(theta[offset:offset + size].reshape(shape))
            offset += size

    return fun, theta0, unpack_to_shared, shared_params, shared_extras


def compile_scipy_diagnostics(
    diagnostic_fn,
    gp_model,
    X_var,
    y_var,
    model=None,
    extra_vars=None,
    extra_init=None,
    frozen_vars=None,
    compile_kwargs=None,
    init="default",
    init_rng=None,
):
    """Compile a diagnostics function that evaluates all namedtuple terms given theta.

    Companion to :func:`compile_scipy_objective`. While that function compiles
    the scalar loss + gradient needed by scipy, this one compiles a cheap
    forward-only pass that returns every term of the namedtuple returned by
    ``diagnostic_fn`` as a numpy scalar. Pass the result to
    :func:`tracked_minimize` to record term histories during optimization.

    Parameters
    ----------
    diagnostic_fn : callable
        ``(gp_model, X_var, y_var) -> namedtuple of TensorVariables``.
        Typically one of the PTGP objectives (``collapsed_elbo``, ``elbo``,
        ``marginal_log_likelihood``) called without the ``.field`` accessor.
    gp_model : GP, VFE, or SVGP
    X_var : TensorVariable
    y_var : TensorVariable
    model : pm.Model, optional
    extra_vars : list of TensorVariable, optional
    extra_init : list of ndarray, optional
    frozen_vars : dict[TensorVariable, ndarray], optional
    compile_kwargs : dict, optional
    init : str
        Same options as :func:`compile_scipy_objective`. Only affects the
        theta layout (shared variable initialization); the returned function
        ignores this layout and evaluates at the theta it receives.
    init_rng : int or numpy Generator, optional

    Returns
    -------
    diag_fn : callable
        ``(theta, X, y) -> namedtuple`` with numpy scalar values.  Same
        namedtuple type as returned by ``diagnostic_fn`` but with float
        values instead of TensorVariables.

    Examples
    --------
    Typical usage with :func:`tracked_minimize`::

        fun, theta0, unpack, shared_params, _ = compile_scipy_objective(
            lambda gp, X, y: collapsed_elbo(gp, X, y).elbo,
            vfe, X_var, y_var, model=model,
        )
        diag_fn = compile_scipy_diagnostics(
            collapsed_elbo, vfe, X_var, y_var, model=model,
        )
        result, history = tracked_minimize(
            fun, theta0, args=(X, y), diag_fn=diag_fn, print_every=10,
        )
        unpack(result.x)
        # history[i] is CollapsedELBOTerms(elbo=..., fit=...,
        #     trace_penalty=..., nystrom_residual=...)
    """
    model = pm.modelcontext(model)

    shared_params, shared_extras, _ = _make_shared_params(
        model, extra_vars, extra_init, init=init, init_rng=init_rng,
        frozen_vars=frozen_vars,
    )

    value_vars_ordered = list(model.continuous_value_vars)
    layout = []
    for vv in value_vars_ordered:
        sv = shared_params[vv]
        val = sv.get_value()
        layout.append((sv, val.shape, val.size))
    if extra_vars is not None:
        for sv in shared_extras:
            val = sv.get_value()
            layout.append((sv, val.shape, val.size))

    theta_var = pt.vector("_theta", dtype="float64")
    pieces = []
    offset = 0
    for _, shape, size in layout:
        pieces.append(theta_var[offset:offset + size].reshape(shape))
        offset += size

    terms = diagnostic_fn(gp_model, X_var, y_var)
    TermsType = type(terms)
    term_list = list(terms)

    [*terms_rvs_replaced] = model.replace_rvs_by_values(term_list)

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

    terms_replaced = [
        graph_replace(t, replace_map, strict=False) for t in terms_rvs_replaced
    ]

    fn = pytensor.function(
        [theta_var, X_var, y_var],
        terms_replaced,
        **(compile_kwargs or {}),
    )

    def diag_fn(theta, X, y):
        values = fn(theta, X, y)
        return TermsType(*[float(v) for v in values])

    return diag_fn


def tracked_minimize(fun, theta0, args, diag_fn=None, print_every=None, **scipy_kwargs):
    """Wrap ``scipy.optimize.minimize`` with optional per-iteration diagnostics.

    At each iteration scipy calls the ``callback``; if ``diag_fn`` is given,
    it is evaluated there and the result appended to ``history``.  The
    callback adds one extra forward pass per iteration (no gradient), which
    is negligible for L-BFGS-B where the Cholesky dominates.

    Parameters
    ----------
    fun : callable
        ``(theta, *args) -> (loss, grad)`` as returned by
        :func:`compile_scipy_objective` (i.e. with ``jac=True`` convention).
    theta0 : ndarray
        Initial parameter vector.
    args : tuple
        Extra arguments forwarded to both ``fun`` and ``diag_fn`` — typically
        ``(X, y)``.
    diag_fn : callable, optional
        ``(theta, *args) -> namedtuple`` from
        :func:`compile_scipy_diagnostics`.  If None, no diagnostics are
        collected and ``history`` is empty.
    print_every : int, optional
        Print a one-line summary every ``print_every`` callback invocations.
        The line shows the iteration count and all namedtuple fields.
        Pass ``None`` (default) for silent operation.
    **scipy_kwargs
        Forwarded to ``scipy.optimize.minimize``.  Defaults applied:
        ``method="L-BFGS-B"``, ``jac=True``.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
    history : list of namedtuples
        One entry per scipy callback invocation (roughly one per iteration
        for L-BFGS-B). Empty if ``diag_fn`` is None.

    Examples
    --------
    ::

        fun, theta0, unpack, shared_params, _ = compile_scipy_objective(
            lambda gp, X, y: collapsed_elbo(gp, X, y).elbo,
            vfe, X_var, y_var, model=model,
        )
        diag_fn = compile_scipy_diagnostics(
            collapsed_elbo, vfe, X_var, y_var, model=model,
        )
        result, history = tracked_minimize(
            fun, theta0, args=(X, y), diag_fn=diag_fn, print_every=10,
        )
        unpack(result.x)

        import matplotlib.pyplot as plt
        plt.plot([-t.elbo for t in history], label="−ELBO")
        plt.plot([t.trace_penalty for t in history], label="trace penalty")
        plt.legend(); plt.show()
    """
    import scipy.optimize

    history = []
    iteration = [0]

    def callback(theta):
        iteration[0] += 1
        if diag_fn is not None:
            terms = diag_fn(theta, *args)
            history.append(terms)
            if print_every is not None and iteration[0] % print_every == 0:
                field_strs = "  ".join(
                    f"{f}={getattr(terms, f):.6g}" for f in type(terms)._fields
                )
                print(f"iter {iteration[0]:5d}  {field_strs}")

    scipy_kwargs.setdefault("method", "L-BFGS-B")
    scipy_kwargs.setdefault("jac", True)
    result = scipy.optimize.minimize(
        fun, theta0, args=args, callback=callback, **scipy_kwargs
    )
    return result, history


def _find_sigma_rv(gp_model, model):
    """Locate the PyMC RV for likelihood sigma in gp_model.

    Handles two cases: sigma is a bare PyMC RV, or wrapped as
    ``pt.assume(sigma_rv, positive=True)``.
    """
    sig = gp_model.likelihood.sigma
    if sig in model.rvs_to_values:
        return sig
    if sig.owner is not None and sig.owner.inputs and sig.owner.inputs[0] in model.rvs_to_values:
        return sig.owner.inputs[0]
    raise ValueError(
        "Cannot identify sigma's PyMC RV from gp_model.likelihood.sigma. "
        "Ensure sigma is a PyMC RV or pt.assume of one registered in the model."
    )


def _staged_build_theta0(model, shared_params, hyper_state,
                         shared_extras=None, Z_state=None):
    """Inject state into shared vars and return the corresponding theta0 vector.

    Writes ``hyper_state`` into ``shared_params`` and optionally ``Z_state``
    into ``shared_extras[0]``, then concatenates all values into the flat
    theta0 vector that matches ``compile_scipy_objective``'s layout.
    """
    pieces = []
    for vv in model.continuous_value_vars:
        val = np.asarray(hyper_state.get(vv, shared_params[vv].get_value()), dtype=np.float64)
        shared_params[vv].set_value(val)
        pieces.append(val.ravel())
    if shared_extras is not None and Z_state is not None:
        shared_extras[0].set_value(Z_state)
        pieces.append(Z_state.ravel())
    return np.concatenate(pieces) if pieces else np.zeros(0)


def minimize_staged_vfe(
    objective_fn,
    gp_model,
    X_var,
    y_var,
    X,
    y,
    model,
    sigma_init,
    Z_var,
    Z_init,
    phase1_freeze_Z=False,
    phase1_maxiter=200,
    phase2_cycles=3,
    phase2_maxiter_Z=100,
    phase2_maxiter_hyper=100,
    phase3_maxiter=300,
    print_every=20,
    compile_kwargs=None,
    scipy_options=None,
    init="default",
    init_rng=None,
):
    """Staged VFE training to prevent inducing point collapse.

    VFE training collapses when the optimizer inflates ``sigma`` to silence
    the trace penalty instead of improving ``Z``. This function blocks that
    failure mode with a structured four-phase schedule:

    1. **Phase 1** — freeze ``sigma`` at ``sigma_init``; train all other
       hyperparameters and (by default) ``Z`` together.
    2. **Phase 2a** — freeze all hyperparameters; train ``Z`` only.
    3. **Phase 2b** — freeze ``Z``; train all hyperparameters (sigma now free).
    4. **Phase 3** — joint fine-tuning of everything (nothing frozen).

    Phases 2a/2b repeat ``phase2_cycles`` times. Pass ``phase2_cycles=0`` to
    skip straight from phase 1 to phase 3 when the ``Z`` initialisation is
    already good (e.g. from ``greedy_variance_init``).

    ``sigma`` is auto-detected from ``gp_model.likelihood.sigma`` — no need to
    pass a separate ``sigma_rv`` argument.

    All diagnostic history entries are :class:`ptgp.objectives.VFEDiagnostics`
    namedtuples, compiled once at function entry with the full theta layout
    (model params + Z). Phase 2b wraps that function to splicing in the
    current ``Z_state``, so the namedtuple type is uniform across all phases.

    Parameters
    ----------
    objective_fn : callable
        ``(gp_model, X_var, y_var) -> scalar``.  Typically
        ``lambda gp, X, y: collapsed_elbo(gp, X, y).elbo``.
    gp_model : VFE
    X_var : TensorVariable
    y_var : TensorVariable
    X : ndarray
    y : ndarray
    model : pm.Model or None
    sigma_init : float
        Constrained-space value to freeze sigma at during phase 1.
    Z_var : TensorVariable
        Symbolic variable for the inducing points (required).
    Z_init : ndarray, shape (M, D)
        Initial inducing point locations.
    phase1_freeze_Z : bool
        If True, freeze ``Z`` during phase 1 as well (useful when ``Z`` is
        pre-initialised from k-means or greedy variance and hyperparameters
        should settle first before Z moves). Default False.
    phase1_maxiter : int
    phase2_cycles : int
    phase2_maxiter_Z : int
    phase2_maxiter_hyper : int
    phase3_maxiter : int
    print_every : int or None
        Print a one-line diagnostic summary every this many scipy iterations.
        Pass None for silent operation.
    compile_kwargs : dict, optional
    scipy_options : dict, optional
        Forwarded as ``options=`` to ``scipy.optimize.minimize``.
    init : str
        Initialisation strategy for phase 1 hyperparameters. Same options as
        :func:`compile_scipy_objective`: ``"default"`` (PyMC initial point — 0
        in unconstrained space), ``"prior_draw"``, or ``"prior_median"``.
        Only affects phase 1; later phases inherit converged values from the
        previous phase.
    init_rng : int or numpy Generator, optional
        Seed for ``"prior_draw"`` / ``"prior_median"``.

    Returns
    -------
    result : scipy.OptimizeResult
        From the final phase-3 minimisation.
    history : list of VFEDiagnostics
        One entry per scipy callback across all phases.
    phase_labels : list of str
        Parallel to ``history``; labels like ``"phase1"``, ``"phase2a_c1"``,
        ``"phase2b_c1"``, ``"phase3"``.
    unpack : callable
        ``(theta) -> None`` from phase 3.  Write ``result.x`` into shared vars
        before calling :func:`compile_predict`.
    shared_params : dict
        From phase 3, for :func:`compile_predict` / :func:`get_trained_params`.
    shared_extras : list
        From phase 3, contains the shared variable for ``Z``.
    """
    model = pm.modelcontext(model)
    sigma_rv = _find_sigma_rv(gp_model, model)
    sigma_vv = model.rvs_to_values[sigma_rv]
    transform = model.rvs_to_transforms.get(sigma_rv)
    if transform is not None:
        sigma_unc = np.asarray(
            transform.forward(pt.as_tensor_variable(float(sigma_init))).eval(),
            dtype=np.float64,
        )
    else:
        sigma_unc = np.asarray(float(sigma_init), dtype=np.float64)
    Z_init_arr = np.asarray(Z_init, dtype=np.float64)

    history = []
    phase_labels = []

    diag_fn_full = compile_scipy_diagnostics(
        vfe_diagnostics, gp_model, X_var, y_var, model=model,
        extra_vars=[Z_var], extra_init=[Z_init_arr],
        compile_kwargs=compile_kwargs,
    )

    def _make_diag_2b(z_snapshot):
        z_flat = z_snapshot.ravel()
        def diag(theta_model, X_, y_):
            return diag_fn_full(np.concatenate([theta_model, z_flat]), X_, y_)
        return diag

    def _run(fun, theta0, unpack, label, maxiter, phase_diag_fn):
        result, ph = tracked_minimize(
            fun, theta0, args=(X, y),
            diag_fn=phase_diag_fn,
            print_every=print_every,
            options={"maxiter": maxiter, **(scipy_options or {})},
        )
        unpack(result.x)
        history.extend(ph)
        phase_labels.extend([label] * len(ph))
        return result

    # Phase 1: freeze sigma (and optionally Z); train everything else
    p1_frozen = {sigma_vv: sigma_unc}
    if phase1_freeze_Z:
        p1_frozen[Z_var] = Z_init_arr
    fun1, theta0_1, unpack1, sp1, se1 = compile_scipy_objective(
        objective_fn, gp_model, X_var, y_var, model=model,
        extra_vars=None if phase1_freeze_Z else [Z_var],
        extra_init=None if phase1_freeze_Z else [Z_init_arr],
        frozen_vars=p1_frozen,
        compile_kwargs=compile_kwargs,
        init=init,
        init_rng=init_rng,
    )
    # When Z is frozen, phase 1 theta has no Z slot; wrap diag_fn_full to
    # append Z_init_arr so its theta layout matches.
    p1_diag = _make_diag_2b(Z_init_arr) if phase1_freeze_Z else diag_fn_full
    _run(fun1, theta0_1, unpack1, "phase1", phase1_maxiter, p1_diag)
    hyper_state = {vv: sp1[vv].get_value().copy() for vv in model.continuous_value_vars}
    Z_state = Z_init_arr.copy() if phase1_freeze_Z else se1[0].get_value().copy()

    # Phase 2: alternate Z-only / hyperparameter-only cycles
    for cycle in range(phase2_cycles):
        # 2a: freeze all hyperparams; train Z
        fun2a, _, unpack2a, sp2a, se2a = compile_scipy_objective(
            objective_fn, gp_model, X_var, y_var, model=model,
            extra_vars=[Z_var], extra_init=[Z_state],
            frozen_vars={vv: hyper_state[vv] for vv in model.continuous_value_vars},
            compile_kwargs=compile_kwargs,
        )
        theta0_2a = _staged_build_theta0(model, sp2a, hyper_state, se2a, Z_state)
        _run(fun2a, theta0_2a, unpack2a, f"phase2a_c{cycle + 1}", phase2_maxiter_Z, diag_fn_full)
        Z_state = se2a[0].get_value().copy()

        # 2b: freeze Z; train all hyperparams (sigma now free)
        fun2b, _, unpack2b, sp2b, _ = compile_scipy_objective(
            objective_fn, gp_model, X_var, y_var, model=model,
            extra_vars=None, extra_init=None,
            frozen_vars={Z_var: Z_state},
            compile_kwargs=compile_kwargs,
        )
        theta0_2b = _staged_build_theta0(model, sp2b, hyper_state)
        _run(fun2b, theta0_2b, unpack2b, f"phase2b_c{cycle + 1}", phase2_maxiter_hyper,
             _make_diag_2b(Z_state))
        hyper_state = {vv: sp2b[vv].get_value().copy() for vv in model.continuous_value_vars}

    # Phase 3: joint fine-tuning
    fun3, _, unpack3, sp3, se3 = compile_scipy_objective(
        objective_fn, gp_model, X_var, y_var, model=model,
        extra_vars=[Z_var], extra_init=[Z_state],
        compile_kwargs=compile_kwargs,
    )
    theta0_3 = _staged_build_theta0(model, sp3, hyper_state, se3, Z_state)
    result = _run(fun3, theta0_3, unpack3, "phase3", phase3_maxiter, diag_fn_full)
    return result, history, phase_labels, unpack3, sp3, se3


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
    compile_kwargs=None,
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
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function``. Use this
        to set ``mode`` (e.g. ``"NUMBA"``, ``"JAX"``), etc. Same pattern as
        ``pm.sample``'s ``compile_kwargs``.

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

    return pytensor.function([X_new_var], [mean_s, var_s], **(compile_kwargs or {}))