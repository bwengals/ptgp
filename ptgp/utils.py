"""Diagnostic utilities for PTGP models."""

import numpy as np


def get_initial_params(model, init="prior_median", rng=None, n_median_samples=500):
    """Return constrained-space initial values for all free RVs in a PyMC model.

    Uses the same initialization strategies as ``compile_scipy_objective``.
    Useful for building proxy kernels with concrete float values to pass to
    ``greedy_variance_init``.

    Parameters
    ----------
    model : pm.Model
    init : str
        ``"default"``, ``"prior_draw"``, or ``"prior_median"``.
    rng : int or numpy Generator, optional
    n_median_samples : int
        Number of draws used to estimate medians. Only used when
        ``init="prior_median"``.

    Returns
    -------
    dict
        ``{rv_name: constrained_value}`` for each free RV.  Scalar RVs are
        returned as plain Python floats; array RVs as numpy arrays.
    """
    import pymc as pm
    from ptgp.optim.training import _make_initial_point

    model = pm.modelcontext(model)
    ip = _make_initial_point(model, init=init, rng=rng, n_median_samples=n_median_samples)

    result = {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        unconstrained = np.asarray(ip[vv.name], dtype=np.float64)
        if transform is not None:
            constrained = np.asarray(
                transform.backward(unconstrained).eval(), dtype=np.float64
            )
        else:
            constrained = unconstrained
        result[rv.name] = float(constrained) if constrained.ndim == 0 else constrained

    return result


_LARGE_GRAD_WARN = 1e4


def check_init(
    fun,
    theta0,
    X,
    y,
    model=None,
    extra_vars=None,
    extra_init=None,
    top_k=10,
):
    """Evaluate the compiled objective at theta0 and report whether the result is finite.

    Prints a one-line summary for the loss and grad, then lists the top ``top_k``
    largest |grad| components annotated with their variable names (when ``model``
    is supplied).

    Parameters
    ----------
    fun : callable
        ``(theta, X, y) -> (loss, grad)`` as returned by
        ``compile_scipy_objective``.
    theta0 : ndarray
        Flat initial parameter vector (the ``theta0`` returned by
        ``compile_scipy_objective``).
    X : ndarray
        Training inputs.
    y : ndarray
        Training targets.
    model : pm.Model, optional
        PyMC model used in ``compile_scipy_objective``.  When given, PyMC
        value vars are labelled by name in the top-K table.
    extra_vars : list of TensorVariable, optional
        Extra symbolic variables passed to ``compile_scipy_objective``.
    extra_init : list of ndarray, optional
        Initial values for ``extra_vars``.  Required for labelling when
        ``extra_vars`` is provided.
    top_k : int
        Number of largest |grad| components to print.  Default 10.

    Returns
    -------
    bool
        True if both loss and all grad components are finite.
    """
    loss, grad = fun(theta0, X, y)
    loss = float(loss)
    grad = np.asarray(grad, dtype=np.float64)

    loss_ok = bool(np.isfinite(loss))
    grad_ok = bool(np.all(np.isfinite(grad)))
    max_g = float(np.abs(grad).max()) if grad.size > 0 else 0.0

    status = "OK" if loss_ok else "NaN/Inf -- BAD"
    print(f"loss at init : {loss:.6g}  ({status})")
    print(f"grad finite  : {grad_ok}  (max |g| = {max_g:.3g})")

    if loss_ok and grad_ok and max_g > _LARGE_GRAD_WARN:
        print(
            f"WARNING: max |grad| = {max_g:.2e} exceeds {_LARGE_GRAD_WARN:.0e}"
            " -- may indicate a pathological initialization"
        )

    labels = _build_index_labels(theta0.size, model, extra_vars, extra_init)

    k = min(top_k, grad.size)
    top_idx = np.argsort(np.abs(grad))[::-1][:k]
    print(f"\ntop-{k} |grad| components:")
    for i in top_idx:
        label = labels[i] if labels is not None else str(i)
        print(f"  [{i:6d}]  {np.abs(grad[i]):.4g}   {label}")

    return loss_ok and grad_ok


def _build_index_labels(n_theta, model, extra_vars, extra_init):
    """Return a list of string labels, one per element of theta.

    Returns None if neither ``model`` nor labelled extras are available.
    """
    try:
        import pymc as pm
    except ImportError:
        return None

    if model is None and extra_vars is None:
        return None

    labels = []

    if model is not None:
        model = pm.modelcontext(model)
        ip = model.initial_point()
        for vv in model.continuous_value_vars:
            val = np.asarray(ip[vv.name])
            name = vv.name
            if val.size == 1:
                labels.append(name)
            else:
                for flat_i in range(val.size):
                    idx = np.unravel_index(flat_i, val.shape)
                    idx_str = ",".join(str(j) for j in idx)
                    labels.append(f"{name}[{idx_str}]")

    if extra_vars is not None and extra_init is not None:
        for var, init_val in zip(extra_vars, extra_init):
            init_val = np.asarray(init_val)
            name = getattr(var, "name", None) or repr(var)
            if init_val.size == 1:
                labels.append(name)
            else:
                for flat_i in range(init_val.size):
                    idx = np.unravel_index(flat_i, init_val.shape)
                    idx_str = ",".join(str(j) for j in idx)
                    labels.append(f"{name}[{idx_str}]")
    elif extra_vars is not None:
        # Names but no shapes -- fill remaining with var name + index
        n_remaining = n_theta - len(labels)
        if n_remaining > 0:
            for var in extra_vars:
                name = getattr(var, "name", None) or repr(var)
                labels.append(f"{name}[?]")

    if len(labels) != n_theta:
        # Shape mismatch -- fall back to plain indices rather than mislabelling.
        return None

    return labels
