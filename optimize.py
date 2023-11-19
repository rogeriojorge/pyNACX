import jax
from jax import jit
import jax.numpy as jnp
from jax import grad
from qsc import Qsc
from time import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from main import nacx_residual

nfp = 2
rc = jnp.array([1, 0.1, 0.01, 0.001])
zs = jnp.array([0, 0.1, 0.01, 0.001])
etabar = 0.9
sigma0=0
iota_desired = 0.4
nphi = 31

@jit
def objective_function(params):
    sigma = jnp.array(params[0:nphi])
    rc = jnp.concatenate([jnp.array([1]),params[nphi:nphi+3]])
    zs = jnp.concatenate([jnp.array([0]),params[nphi+3:nphi+6]])
    etabar = params[-2]
    iota = params[-1]
    residuals, elongation = nacx_residual(eR=rc, eZ=zs, etabar=etabar, sigma=sigma, sigma0=sigma0, iota=iota, nphi=nphi)
    return 1e3*jnp.sum(residuals**2)/nphi + 1e-3*jnp.sum(elongation**2)/nphi + 1e3*(iota-iota_desired)**2

print('Do optimization')
zs=zs[1:]
rc=rc[1:]
sigma = jnp.zeros(nphi)
initial_params = jnp.concatenate([sigma,rc,zs,jnp.array([etabar,iota_desired])])
Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
print('Initial objective function: {}'.format(objective_function(initial_params)))

from scipy.optimize import minimize
start_time = time()
result = minimize(objective_function, initial_params, method='BFGS', jac=jax.grad(objective_function), options={'disp':True, 'maxiter':1e4})
print('Optimization took {} seconds'.format(time() - start_time))
# import numpy as np
from scipy.optimize import least_squares
# def objective_function_jac(params):
#     return np.array(jax.jacfwd(objective_function)(params))
# def objective_function_np(params):
#     return np.array(objective_function(params))
# result = least_squares(objective_function_np, initial_params, jac=objective_function_jac, verbose=2)#, method='lm', verbose=2)
optimized_params = result.x

# from jax.scipy.optimize import minimize
# result = minimize(objective_function, initial_params, method="BFGS")
# optimized_params = result.x

# import jaxopt
# tol_optimization=1e-6
# max_nfev_optimization=10000
# optimizer = jaxopt.ScipyMinimize(fun=objective_function, method='L-BFGS-B', tol=tol_optimization, maxiter=max_nfev_optimization, jit=True)#, options={'jac':True})
# optimized_params, state = optimizer.run(initial_params)

optimized_sigma, optimized_rc, optimized_zs, optimized_etabar, optimized_iota = optimized_params[0:nphi], optimized_params[nphi:nphi+3], optimized_params[nphi+3:nphi+6], optimized_params[-2], optimized_params[-1]
print('Optimized rc: {}'.format(optimized_rc))
print('Optimized zs: {}'.format(optimized_zs))
print('Optimized etabar: {}'.format(optimized_etabar))
print('Optimized iota: {}'.format(optimized_iota))
print('Optimized objective function: {}'.format(objective_function(optimized_params)))
objective_function(optimized_params)
objective_function(optimized_params)
rc = jnp.concatenate([jnp.array([1]), optimized_rc])
zs = jnp.concatenate([jnp.array([0]), optimized_zs])
etabar = optimized_etabar
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
print(f'True iota: {stel.iota}')
# stel.plot()
# stel.plot_boundary(r=0.1)
exit()