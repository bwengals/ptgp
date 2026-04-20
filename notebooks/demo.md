---
jupyter:
  jupytext:
    formats: md,ipynb
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

Test all three models (GP, VFE, SVGP) on synthetic 1D regression data using PyMC priors.

```python
import numpy as np
import pymc as pm
import preliz as pz
import pytensor
import pytensor.tensor as pt
import matplotlib.pyplot as plt

import ptgp as pg
```

## Generate data, student-t noise

```python
rng = np.random.default_rng(600)
N = 200
noise_std = 0.2
nu_true = 6

# True GP hyperparameters
eta_true = 1.3
ls_true = 1.0

X_train = np.sort(rng.uniform(0, 10, N))[:, None]
X_test = np.linspace(-1, 11, 200)[:, None]

# Draw from a GP prior with Matern-5/2 kernel
K = eta_true**2 * pg.Matern52(input_dim=1, ls=ls_true)
K = K(X_train, X_train).eval() + 1e-6 * np.eye(N)

f_train = rng.multivariate_normal(np.zeros(N), K)
y_train = f_train + noise_std * rng.standard_t(df=nu_true, size=N)

plt.scatter(X_train, y_train, s=10, label="data")
plt.plot(X_train, f_train, "k--", alpha=0.5, label="true f")
plt.legend()
plt.title("Training data");
```

## Exact GP

```python
X_var = pt.matrix("X")
y_var = pt.vector("y")

with pm.Model() as gp_model:
    ls = pm.HalfFlat("ls")
    eta = pm.Exponential("eta", scale=2.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    kernel = eta**2 * pg.Matern52(input_dim=1, ls=ls)
    gp = pg.GP(kernel=kernel, likelihood=pg.Gaussian(sigma=sigma))

train_step, shared_params, shared_extras = pg.compile_training_step(
    pg.marginal_log_likelihood, gp, X_var, y_var,
    pm_model=gp_model, learning_rate=1e-2,
)

for i in range(500):
    loss = train_step(X_train, y_train)
    if i % 100 == 0:
        print(f"Step {i}: loss = {loss:.4f}")

params = pg.get_trained_params(gp_model, shared_params)
print(f"\nRecovered: {params}")
print(f"True:      eta={eta_true}, ls={ls_true}, noise_std={noise_std}")
```

```python
X_new_var = pt.matrix("X_new")
predict_gp = pg.compile_predict(
    gp, X_new_var, gp_model, shared_params,
    X_train=X_train, y_train=y_train,
)
predict_gp_y = pg.compile_predict(
    gp, X_new_var, gp_model, shared_params,
    X_train=X_train, y_train=y_train, incl_lik=True,
)

mu_gp, var_gp = predict_gp(X_test)
_, var_gp_y = predict_gp_y(X_test)
sd_gp = np.sqrt(var_gp)
sd_gp_y = np.sqrt(var_gp_y)

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_gp, "C0", label="GP mean")
plt.fill_between(X_test.ravel(), mu_gp - 2*sd_gp_y, mu_gp + 2*sd_gp_y, alpha=0.1, color="C0", label="y 2 sd")
plt.fill_between(X_test.ravel(), mu_gp - 2*sd_gp, mu_gp + 2*sd_gp, alpha=0.3, color="C0", label="f 2 sd")
plt.plot(X_train, f_train, "k--", alpha=0.3, label="true f")
plt.legend()
plt.title("Exact GP");
```

## VFE (Sparse GP -- collapsed bound)

```python
M = 20
Z_init = np.linspace(-0.5, 10.5, M)[:, None]

Z_var = pt.matrix("Z")

with pm.Model() as vfe_model:
    ls = pm.HalfFlat("ls")
    eta = pm.Exponential("eta", lam=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    vfe = pg.VFE(
        kernel=eta**2 * pg.Matern52(input_dim=1, ls=ls),
        likelihood=pg.Gaussian(sigma=sigma),
        inducing_variable=pg.InducingPoints(Z_var),
    )

train_step_vfe, shared_params_vfe, shared_extras_vfe = pg.compile_training_step(
    pg.collapsed_elbo, vfe, X_var, y_var,
    pm_model=vfe_model,
    extra_vars=[Z_var],
    extra_init=[Z_init],
    learning_rate=1e-2,
)

for i in range(500):
    loss = train_step_vfe(X_train, y_train)
    if i % 100 == 0:
        print(f"Step {i}: loss = {loss:.4f}")

params_vfe = pg.get_trained_params(vfe_model, shared_params_vfe)
print(f"\nRecovered: {params_vfe}")
print(f"True:      eta={eta_true}, ls={ls_true}, noise_std={noise_std}")
```

```python
predict_vfe = pg.compile_predict(
    vfe, X_new_var, vfe_model, shared_params_vfe,
    X_train=X_train, y_train=y_train,
    extra_vars=[Z_var],
    shared_extras=shared_extras_vfe,
)
predict_vfe_y = pg.compile_predict(
    vfe, X_new_var, vfe_model, shared_params_vfe,
    X_train=X_train, y_train=y_train,
    extra_vars=[Z_var],
    shared_extras=shared_extras_vfe, incl_lik=True,
)

mu_vfe, var_vfe = predict_vfe(X_test)
_, var_vfe_y = predict_vfe_y(X_test)
sd_vfe = np.sqrt(var_vfe)
sd_vfe_y = np.sqrt(var_vfe_y)

Z_final = shared_extras_vfe[0].get_value()

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_vfe, "C0", label="VFE mean")
plt.fill_between(X_test.ravel(), mu_vfe - 2*sd_vfe_y, mu_vfe + 2*sd_vfe_y, alpha=0.1, color="C0", label="y 2 sd")
plt.fill_between(X_test.ravel(), mu_vfe - 2*sd_vfe, mu_vfe + 2*sd_vfe, alpha=0.3, color="C0", label="f 2 sd")
plt.scatter(Z_final, np.zeros(M) - 2, marker="^", c="r", s=40, label="inducing pts")
plt.plot(X_train, f_train, "k--", alpha=0.3, label="true f")
plt.legend(loc="upper right")
plt.title(f"VFE (M={M})");
```

## SVGP (Stochastic Variational GP)

Minibatch training with variational parameters.

```python
M_svgp = 20
Z_svgp = np.linspace(-0.5, 10.5, M_svgp)[:, None]

q_mu_var = pt.vector("q_mu")
q_sqrt_var = pt.matrix("q_sqrt")
Z_var = pt.matrix("Z")

with pm.Model() as svgp_model:
    ls = pm.HalfFlat("ls")
    eta = pm.Exponential("eta", lam=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    nu = pm.Gamma("nu", alpha=2, beta=0.1)

    svgp = pg.SVGP(
        kernel=eta**2 * pg.Matern52(input_dim=1, ls=ls),
        likelihood=pg.StudentT(sigma=sigma, nu=nu),
        inducing_variable=pg.InducingPoints(Z_var),
        q_mu=q_mu_var,
        q_sqrt=q_sqrt_var,
    )

X_batch_var = pt.matrix("X_batch")
y_batch_var = pt.vector("y_batch")

def elbo_scaled(model, X, y):
    return pg.elbo(model, X, y, n_data=N)

train_step_svgp, shared_params_svgp, shared_extras_svgp = pg.compile_training_step(
    elbo_scaled, svgp, X_batch_var, y_batch_var,
    pm_model=svgp_model,
    extra_vars=[q_mu_var, q_sqrt_var, Z_var],
    extra_init=[np.zeros(M_svgp), np.eye(M_svgp), Z_svgp],
    learning_rate=1e-2,
)

batch_size = 32
losses = []
for step in range(2000):
    idx = rng.choice(N, size=batch_size, replace=False)
    loss = train_step_svgp(X_train[idx], y_train[idx])
    losses.append(float(loss))
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss:.2f}")

params_svgp = pg.get_trained_params(svgp_model, shared_params_svgp)
print(f"\nRecovered: {params_svgp}")
print(f"True:      eta={eta_true}, ls={ls_true}, noise_std={noise_std}, nu={nu_true}")
```

```python
plt.figure(figsize=(10, 3))
plt.plot(losses)
plt.xlabel("step")
plt.ylabel("neg ELBO")
plt.title("SVGP training loss");
```

```python
predict_svgp = pg.compile_predict(
    svgp, X_new_var, svgp_model, shared_params_svgp,
    extra_vars=[q_mu_var, q_sqrt_var, Z_var],
    shared_extras=shared_extras_svgp,
)
predict_svgp_y = pg.compile_predict(
    svgp, X_new_var, svgp_model, shared_params_svgp,
    extra_vars=[q_mu_var, q_sqrt_var, Z_var],
    shared_extras=shared_extras_svgp, incl_lik=True,
)

mu_svgp, var_svgp = predict_svgp(X_test)
mu_svgp_y, var_svgp_y = predict_svgp_y(X_test)
sd_svgp = np.sqrt(var_svgp)
sd_svgp_y = np.sqrt(var_svgp_y)

plt.figure(figsize=(10, 4))
plt.scatter(X_train, y_train, s=10, c="k", zorder=3, label="data")
plt.plot(X_test, mu_svgp, "C0", label="SVGP mean")
plt.fill_between(X_test.ravel(), mu_svgp_y - 2*sd_svgp_y, mu_svgp_y + 2*sd_svgp_y, alpha=0.1, color="C0", label="y 2 sd")
plt.fill_between(X_test.ravel(), mu_svgp - 2*sd_svgp, mu_svgp + 2*sd_svgp, alpha=0.3, color="C0", label="f 2 sd")
Z_opt = shared_extras_svgp[2].get_value()
plt.scatter(Z_opt, np.zeros(M_svgp) - 2, marker="^", c="r", s=40, label="inducing pts (optimized)")
plt.plot(X_train, f_train, "k--", alpha=0.3, label="true f")
plt.legend()
plt.title(f"SVGP (M={M_svgp})");
```

## Comparison

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

for ax, (mu, sd_f, sd_y, title) in zip(axes, [
    (mu_gp, sd_gp, sd_gp_y, "Exact GP"),
    (mu_vfe, sd_vfe, sd_vfe_y, f"VFE (M={M})"),
    (mu_svgp, sd_svgp, sd_svgp_y, f"SVGP (M={M_svgp})"),
]):
    ax.scatter(X_train, y_train, s=8, c="k", zorder=3)
    ax.plot(X_test, mu, "C0")
    ax.fill_between(X_test.ravel(), mu - 2*sd_y, mu + 2*sd_y, alpha=0.1, color="C0")
    ax.fill_between(X_test.ravel(), mu - 2*sd_f, mu + 2*sd_f, alpha=0.3, color="C0")
    ax.plot(X_train, f_train, "k--", alpha=0.3)
    ax.set_title(title)
    ax.set_xlim(-1, 11)

fig.tight_layout();
```

```python

```
