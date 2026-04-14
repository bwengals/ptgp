---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: ptgp
    language: python
    name: python3
---

# PTGP Demo

Test all three models (GP, VFE, SVGP) on synthetic 1D regression data.

```python
import numpy as np
import pytensor
import pytensor.tensor as pt
import jax.numpy as jnp
import matplotlib.pyplot as plt

import ptgp as pg
```

## Generate data

```python
rng = np.random.default_rng(42)
N = 100
X_train = np.sort(rng.uniform(0, 10, N))[:, None]
y_train = np.sin(X_train.ravel()) + 0.2 * rng.standard_normal(N)

X_test = np.linspace(-1, 11, 200)[:, None]

plt.scatter(X_train, y_train, s=10, label="data")
plt.plot(X_test, np.sin(X_test), "k--", alpha=0.5, label="true")
plt.legend()
plt.title("Training data");
```

## Exact GP

Optimize lengthscale, amplitude, and noise with BFGS.

Note: `pm.Model()` context with named priors is the intended pattern (see DESIGN.md), but PyMC currently can't import with the assumption-system PyTensor branch. Using named unconstrained variables with `pt.exp` as a stand-in.

```python
# Unconstrained parameters (stand-in for pm.LogNormal / pm.HalfNormal priors)
log_ls = pt.scalar("log_ls")      # ls = pm.LogNormal("ls", mu=0, sigma=1)
log_eta = pt.scalar("log_eta")    # eta = pm.HalfNormal("eta", sigma=1)
log_sigma = pt.scalar("log_sigma")  # sigma = pm.HalfNormal("sigma", sigma=1)

ls = pt.exp(log_ls)
eta = pt.exp(log_eta)
sigma = pt.exp(log_sigma)

kernel = eta**2 * pg.Matern52(ls=ls)
gp = pg.GP(kernel=kernel, likelihood=pg.Gaussian(sigma=sigma))

opt_values, result = pg.fit_bfgs(
    pg.marginal_log_likelihood, gp, X_train, y_train,
    params=[log_ls, log_eta, log_sigma],
    init_values=[0.0, 0.0, -1.0],
)

ls_opt = np.exp(opt_values[0])
eta_opt = np.exp(opt_values[1])
sigma_opt = np.exp(opt_values[2])
print(f"Optimized ls={ls_opt:.4f}, eta={eta_opt:.4f}, sigma={sigma_opt:.4f}")
print(f"Neg-MLL: {result.fun:.4f}")
```

```python
# Predict with optimized hyperparameters
gp_opt = pg.GP(
    kernel=eta_opt**2 * pg.Matern52(ls=ls_opt),
    likelihood=pg.Gaussian(sigma=sigma_opt),
)
fmean, fvar = gp_opt.predict_f(
    pt.as_tensor_variable(X_test),
    pt.as_tensor_variable(X_train),
    pt.as_tensor_variable(y_train),
)
f = pytensor.function([], [fmean, fvar])
mu_gp, var_gp = f()
sd_gp = np.sqrt(var_gp)

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_gp, label="GP mean")
plt.fill_between(X_test.ravel(), mu_gp - 2*sd_gp, mu_gp + 2*sd_gp, alpha=0.2, label="GP 2 sd")
plt.plot(X_test, np.sin(X_test), "k--", alpha=0.3, label="true")
plt.legend()
plt.title("Exact GP");
```

## VFE (Sparse GP -- collapsed bound)

Optimize hyperparameters with inducing points fixed.

```python
M = 15
Z_init = np.linspace(0.5, 9.5, M)[:, None]

log_ls_v = pt.scalar("log_ls_v")
log_eta_v = pt.scalar("log_eta_v")
log_sigma_v = pt.scalar("log_sigma_v")

vfe = pg.VFE(
    kernel=pt.exp(log_eta_v)**2 * pg.Matern52(ls=pt.exp(log_ls_v)),
    likelihood=pg.Gaussian(sigma=pt.exp(log_sigma_v)),
    inducing_variable=pg.InducingPoints(pt.as_tensor_variable(Z_init)),
)

opt_vals_vfe, res_vfe = pg.fit_bfgs(
    pg.collapsed_elbo, vfe, X_train, y_train,
    params=[log_ls_v, log_eta_v, log_sigma_v],
    init_values=[0.0, 0.0, -1.0],
)

ls_vfe = np.exp(opt_vals_vfe[0])
eta_vfe = np.exp(opt_vals_vfe[1])
sigma_vfe = np.exp(opt_vals_vfe[2])
print(f"VFE optimized ls={ls_vfe:.4f}, eta={eta_vfe:.4f}, sigma={sigma_vfe:.4f}")
print(f"Neg collapsed-ELBO: {res_vfe.fun:.4f}")
```

```python
# VFE prediction
vfe_opt = pg.VFE(
    kernel=eta_vfe**2 * pg.Matern52(ls=ls_vfe),
    likelihood=pg.Gaussian(sigma=sigma_vfe),
    inducing_variable=pg.InducingPoints(pt.as_tensor_variable(Z_init)),
)
fmean_v, fvar_v = vfe_opt.predict_f(
    pt.as_tensor_variable(X_test),
    pt.as_tensor_variable(X_train),
    pt.as_tensor_variable(y_train),
)
f_vfe = pytensor.function([], [fmean_v, fvar_v])
mu_vfe, var_vfe = f_vfe()
sd_vfe = np.sqrt(var_vfe)

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_vfe, label="VFE mean")
plt.fill_between(X_test.ravel(), mu_vfe - 2*sd_vfe, mu_vfe + 2*sd_vfe, alpha=0.2, label="VFE 2 sd")
plt.scatter(Z_init, np.zeros(M) - 2, marker="^", c="r", s=40, label="inducing pts")
plt.plot(X_test, np.sin(X_test), "k--", alpha=0.3, label="true")
plt.legend()
plt.title(f"VFE (M={M})");
```

## SVGP (Stochastic Variational GP)

Minibatch SGD with user-controlled training loop.

```python
import optax

M_svgp = 15
Z_svgp = np.linspace(0.5, 9.5, M_svgp)[:, None]

# Symbolic parameters
log_ls_s = pt.scalar("log_ls_s")
log_eta_s = pt.scalar("log_eta_s")
log_sigma_s = pt.scalar("log_sigma_s")
q_mu_var = pt.vector("q_mu")
q_sqrt_var = pt.matrix("q_sqrt")

svgp = pg.SVGP(
    kernel=pt.exp(log_eta_s)**2 * pg.Matern52(ls=pt.exp(log_ls_s)),
    likelihood=pg.Gaussian(sigma=pt.exp(log_sigma_s)),
    inducing_variable=pg.InducingPoints(pt.as_tensor_variable(Z_svgp)),
    q_mu=q_mu_var,
    q_sqrt=q_sqrt_var,
)

X_var = pt.matrix("X_batch")
y_var = pt.vector("y_batch")

def elbo_scaled(model, X, y):
    return pg.elbo(model, X, y, n_data=N)

params = [log_ls_s, log_eta_s, log_sigma_s, q_mu_var, q_sqrt_var]
optimizer = optax.adam(1e-2)

init_fn, step_fn = pg.make_training_step(
    elbo_scaled, svgp, X_var, y_var,
    params=params, optimizer=optimizer,
)

# Initialize parameter values
param_values = (
    jnp.array(0.0),       # log_ls
    jnp.array(0.0),       # log_eta
    jnp.array(-1.0),      # log_sigma
    jnp.zeros(M_svgp),    # q_mu
    jnp.eye(M_svgp),      # q_sqrt
)
opt_state = init_fn(param_values)

# Training loop
batch_size = 32
losses = []
for step in range(2000):
    idx = rng.choice(N, size=batch_size, replace=False)
    X_batch = jnp.array(X_train[idx])
    y_batch = jnp.array(y_train[idx])
    param_values, opt_state, loss_val = step_fn(X_batch, y_batch, param_values, opt_state)
    losses.append(float(loss_val))
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss_val:.2f}")

ls_s = np.exp(float(param_values[0]))
eta_s = np.exp(float(param_values[1]))
sigma_s = np.exp(float(param_values[2]))
print(f"Final: ls={ls_s:.4f}, eta={eta_s:.4f}, sigma={sigma_s:.4f}")
```

```python
plt.figure(figsize=(10, 3))
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("neg ELBO")
plt.title("SVGP training loss");
```

```python
# SVGP prediction with optimized params
svgp_opt = pg.SVGP(
    kernel=eta_s**2 * pg.Matern52(ls=ls_s),
    likelihood=pg.Gaussian(sigma=sigma_s),
    inducing_variable=pg.InducingPoints(pt.as_tensor_variable(Z_svgp)),
    q_mu=pt.as_tensor_variable(np.asarray(param_values[3])),
    q_sqrt=pt.as_tensor_variable(np.asarray(param_values[4])),
)

fmean_s, fvar_s = svgp_opt.predict_f(pt.as_tensor_variable(X_test))
f_svgp = pytensor.function([], [fmean_s, fvar_s])
mu_svgp, var_svgp = f_svgp()
sd_svgp = np.sqrt(var_svgp)

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_svgp, label="SVGP mean")
plt.fill_between(X_test.ravel(), mu_svgp - 2*sd_svgp, mu_svgp + 2*sd_svgp, alpha=0.2, label="SVGP 2 sd")
plt.scatter(Z_svgp, np.zeros(M_svgp) - 2, marker="^", c="r", s=40, label="inducing pts")
plt.plot(X_test, np.sin(X_test), "k--", alpha=0.3, label="true")
plt.legend()
plt.title(f"SVGP (M={M_svgp})");
```

## Comparison

Overlay all three model predictions.

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

for ax, (mu, sd, title) in zip(axes, [
    (mu_gp, sd_gp, "Exact GP"),
    (mu_vfe, sd_vfe, f"VFE (M={M})"),
    (mu_svgp, sd_svgp, f"SVGP (M={M_svgp})"),
]):
    ax.scatter(X_train, y_train, s=8, c="k", zorder=3)
    ax.plot(X_test, mu, "C0")
    ax.fill_between(X_test.ravel(), mu - 2*sd, mu + 2*sd, alpha=0.2, color="C0")
    ax.plot(X_test, np.sin(X_test), "k--", alpha=0.3)
    ax.set_title(title)
    ax.set_xlim(-1, 11)

fig.tight_layout();
```
