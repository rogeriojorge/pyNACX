#!/usr/bin/env python3
import re
import jax
import json
from jax import jit
import jax.numpy as jnp
from flax import serialization
import matplotlib.pyplot as plt
import scipy
from train_inn import DeepNN, number_of_x_parameters, forward_solver, eReZmin, eReZmax, etabarMin, etabarMax, model_save_path, loss_save_path

rc_base = 0.07
zs_base = 0.07
eta_base = 0.9
n_samples_test = 41

nfp = int(re.search('nfp(\d+)', model_save_path).group(1))
print(f'nfp = {nfp}')

model = DeepNN()
dummy_input = jnp.ones((1, number_of_x_parameters))
init_params = model.init(jax.random.PRNGKey(0), dummy_input)
with open(model_save_path, 'rb') as f:
    bytes_params = f.read()
params = serialization.from_bytes(init_params, bytes_params)
with open(loss_save_path, 'r') as f:
    loaded_loss_array = json.load(f)
loss_array = jnp.array(loaded_loss_array)

@jit
def objective(x, rc=rc_base, zs=zs_base, eta=eta_base):
    return jnp.square(model.apply(params, x)-jnp.array([rc, zs, eta]))
@jit
def gradient(x, rc=rc_base, zs=zs_base, eta=eta_base):
    return jax.jacfwd(objective, argnums=0)(x, rc, zs, eta)

## Vary rc
rc_array = jnp.linspace(eReZmin, eReZmax, n_samples_test)
iota_true_rc, elongation_true_rc, iL_true_rc = jnp.array([forward_solver([er, zs_base, eta_base]) for er in rc_array]).T
iota_predicted_rc, elongation_predicted_rc, iL_predicted_rc = jnp.array([scipy.optimize.root(objective, x0=jnp.array([iota_true_rc[i], elongation_true_rc[i], iL_true_rc[i]]),
                                                                                             method='lm', jac=gradient, args=(er, zs_base, eta_base), tol=1e-9, options={"ftol": 1e-9, "maxiter":10000}).x
                                                                                             for i, er in enumerate(rc_array)]).T
## Vary zs
zs_array = jnp.linspace(eReZmin, eReZmax, n_samples_test)
iota_true_zs, elongation_true_zs, iL_true_zs = jnp.array([forward_solver([rc_base, zs, eta_base]) for zs in zs_array]).T
iota_predicted_zs, elongation_predicted_zs, iL_predicted_zs = jnp.array([scipy.optimize.root(objective, x0=jnp.array([iota_true_zs[i], elongation_true_zs[i], iL_true_zs[i]]),
                                                                                             method='lm', jac=gradient, args=(rc_base, zs, eta_base), tol=1e-9, options={"ftol": 1e-9, "maxiter":10000}).x
                                                                                             for i, zs in enumerate(zs_array)]).T
## Vary eta
eta_array = jnp.linspace(etabarMin, etabarMax, n_samples_test)
iota_true_eta, elongation_true_eta, iL_true_eta = jnp.array([forward_solver([rc_base, zs_base, eta]) for eta in eta_array]).T
iota_predicted_eta, elongation_predicted_eta, iL_predicted_eta = jnp.array([scipy.optimize.root(objective, x0=jnp.array([iota_true_eta[i], elongation_true_eta[i], iL_true_eta[i]]),
                                                                                             method='lm', jac=gradient, args=(rc_base, zs_base, eta), tol=1e-9, options={"ftol": 1e-9, "maxiter":10000}).x
                                                                                             for i, eta in enumerate(eta_array)]).T

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# Plot for loss
axs[0, 0].semilogy(loss_array)
axs[0, 0].set(xlabel='training step', ylabel='loss')
# Plot for iota
axs[0, 1].plot(rc_array, iota_predicted_rc, 'r.',  label='NN  rc')
axs[0, 1].plot(rc_array, iota_true_rc, 'r--',      label='qsc rc')
axs[0, 1].plot(rc_array, iota_predicted_zs, 'b.',  label='NN  zs')
axs[0, 1].plot(rc_array, iota_true_zs, 'b--',      label='qsc zs')
axs[0, 1].plot(rc_array, iota_predicted_eta, 'k.', label='NN  eta')
axs[0, 1].plot(rc_array, iota_true_eta, 'k--',     label='qsc eta')
axs[0, 1].set(xlabel='rc/zs/eta', ylabel='iota')
axs[0, 1].legend()
# Plot for elongation
axs[1, 0].plot(rc_array, elongation_predicted_rc, 'r.',  label='NN  rc')
axs[1, 0].plot(rc_array, elongation_true_rc, 'r--',      label='qsc rc')
axs[1, 0].plot(rc_array, elongation_predicted_zs, 'b.',  label='NN  zs')
axs[1, 0].plot(rc_array, elongation_true_zs, 'b--',      label='qsc zs')
axs[1, 0].plot(rc_array, elongation_predicted_eta, 'k.', label='NN  eta')
axs[1, 0].plot(rc_array, elongation_true_eta, 'k--',     label='qsc eta')
axs[1, 0].set(xlabel='rc/zs/eta', ylabel='max elongation')
axs[1, 0].legend()
# Plot for iL
axs[1, 1].plot(rc_array, iL_predicted_rc, 'r.',  label='NN  rc')
axs[1, 1].plot(rc_array, iL_true_rc, 'r--',      label='qsc rc')
axs[1, 1].plot(rc_array, iL_predicted_zs, 'b.',  label='NN  zs')
axs[1, 1].plot(rc_array, iL_true_zs, 'b--',      label='qsc zs')
axs[1, 1].plot(rc_array, iL_predicted_eta, 'k.', label='NN  eta')
axs[1, 1].plot(rc_array, iL_true_eta, 'k--',     label='qsc eta')
axs[1, 1].set(xlabel='rc/zs/eta', ylabel='max inverse L gradB')
axs[1, 1].legend()
#
plt.tight_layout()
plt.show()