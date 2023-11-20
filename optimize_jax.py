import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import jit
import jax.numpy as jnp
from jax import grad
from qsc import Qsc
from time import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from main_jax import nacx_residual

nfp = 2
nfourier = 6
etabar = 0.9
iota_min = 0.41
nphi = 71

rc_in = jnp.power(10.0, -jnp.arange(nfourier))
zs_in = jnp.concatenate([jnp.array([0]), jnp.power(10.0, -jnp.arange(1, nfourier))])

@jit
def objective_function(params):
    length = params.size
    etabar = params[-1]
    length_of_each = (length - 1) // 2
    rc = jnp.concatenate([jnp.array([1]), params[:length_of_each]])
    zs = jnp.concatenate([jnp.array([0]), params[length_of_each:-1]])
    iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
    # return 0 \
    #      + jnp.sum(inv_L_grad_B**2)/nphi \
    #      + jnp.sum(elongation**2)/nphi \
    #      + 1e2*(jnp.abs(jnp.max(jnp.array([jnp.array([0]),jnp.array([(1-jnp.abs(iota)/jnp.abs(iota_min))])]))))**2 \
    # return jnp.concatenate([inv_L_grad_B, elongation, jnp.array([1e3*jnp.abs(jnp.min(jnp.array([jnp.array([0]),jnp.array([(jnp.abs(iota)-jnp.abs(iota_min))])])))])])
    return jnp.concatenate([inv_L_grad_B, elongation, 1e2*jnp.array([jnp.abs(jnp.max(jnp.array([jnp.array([0]),jnp.array([(1-jnp.abs(iota)/jnp.abs(iota_min))])])))])])

@jit
def grad_objective_function(params):
    grad_func = jax.jacfwd(objective_function)(params)
    return grad_func

# @jit
# def hess_objective_function(params):
#     hess_func = jax.jacfwd(grad_objective_function)(params)
#     return hess_func

print("Optimizing with {} fourier modes".format(nfourier))
initial_params = jnp.concatenate([rc_in[1:],zs_in[1:],jnp.array([etabar])])
print("First jit'ing objective function and gradient")
start_time=time();print(f'Initial sum of objective function: {jnp.sum(objective_function(initial_params)**2):.1e} took {(time() - start_time):.1f} seconds')
start_time=time();print(f'Initial sum of grad of objective function: {jnp.sum(grad_objective_function(initial_params)**2):.1e} took {(time() - start_time):.1f} seconds')
# start_time=time();print(f'Initial sum of hess of objective function: {jnp.sum(hess_objective_function(initial_params)**2):.1e} took {(time() - start_time):.1f} seconds')

# from scipy.optimize import minimize
# start_time = time()
# result = minimize(objective_function, initial_params, method='Newton-CG', jac=grad_objective_function, hess=hess_objective_function, options={'disp':True, 'maxiter':1e3})
# # result = minimize(objective_function, initial_params, method='BFGS', jac=grad_objective_function, options={'disp':True, 'maxiter':1e3})
# print('Final objective function {} took {} seconds'.format(result.fun, time() - start_time))

import numpy as np
from scipy.optimize import least_squares
def objective_function_jac(params):
    return np.array(grad_objective_function(params))
def objective_function_np(params):
    return np.array(objective_function(params))
start_time = time()
result = least_squares(objective_function_np, initial_params, jac=objective_function_jac, verbose=2, method='lm', x_scale='jac', max_nfev=int(5e3))
print('Optimization took {} seconds'.format(time() - start_time))

# from jax.scipy.optimize import minimize
# result = minimize(objective_function, initial_params, method="BFGS")

optimized_params = result.x

# import jaxopt
# tol_optimization=1e-6
# max_nfev_optimization=10000
# optimizer = jaxopt.ScipyMinimize(fun=objective_function, method='L-BFGS-B', tol=tol_optimization, maxiter=max_nfev_optimization, jit=True)#, options={'jac':True})
# optimized_params, state = optimizer.run(initial_params)

rc = jnp.concatenate([jnp.array([1]), optimized_params[0:len(rc_in)-1]])
zs = jnp.concatenate([jnp.array([0]), optimized_params[len(rc_in)-1:len(rc_in)+len(zs_in)-2]])
etabar = optimized_params[-1]
iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
print('Optimized rc: {}'.format(rc))
print('Optimized zs: {}'.format(zs))
print('Optimized etabar: {}'.format(etabar))
print('Optimized iota: {}'.format(iota))
print('Optimized min inv_L_grad_B: {}'.format(jnp.min(inv_L_grad_B)))
print('Optimized max elongation: {}'.format(jnp.max(elongation)))
print(f'True iota: {stel.iota}')
print(f'True min inv_L_grad_B: {jnp.min(stel.inv_L_grad_B)}')
print(f'True max elongation: {jnp.max(stel.elongation)}')
# stel.plot()
# stel.plot_boundary(r=0.1)
exit()