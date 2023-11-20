import jax
jax.default_device(jax.devices("cpu")[0])
from jax import jit, lax
import jax.numpy as jnp
from qsc import Qsc
from time import time
from functools import partial
import matplotlib.pyplot as plt

nfp = 2
rc = jnp.array([1, -0.1, 0.01, 0.001])
zs = jnp.array([0, 0.1, 0.01, 0.001])
sG = 1
spsi = 1
I2 = 0
B0 = 1
etabar = 0.9
sigma0=0
iota_desired = 0.4
nphi = 151

@partial(jit, static_argnums=(5,6,7))
def nacx_residual(eR, eZ, etabar=1.0, sigma=jnp.zeros(nphi)+0.01, iota=iota_desired, nphi=nphi, sigma0=sigma0, debug=False):
    assert nphi % 2 == 1, 'nphi must be odd'
    phi_vals = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    d_phi = phi_vals[1] - phi_vals[0]

    @jit
    def replace_first_element(x, new_value):
        return jnp.concatenate([jnp.array([new_value]), x[1:]])
    
    sigma = replace_first_element(sigma, sigma0)

    @jit
    def pos_vector_component(phi_val):
        rc_cosines = jnp.cos(jnp.arange(eR.size) * phi_val * nfp)
        zs_sines   = jnp.sin(jnp.arange(eZ.size) * phi_val * nfp)
        R = jnp.sum(eR * rc_cosines)
        Z = jnp.sum(eZ * zs_sines)
        return jnp.array([R * jnp.cos(phi_val), R * jnp.sin(phi_val), Z])

    @jit
    def d_r_d_phi_func(phi_val):
        return jax.jacfwd(pos_vector_component)(phi_val)

    @jit
    def d2_r_d_phi2_func(phi_val):
        return jax.jacfwd(d_r_d_phi_func)(phi_val)

    @jit
    def d3_r_d_phi3_func(phi_val):
        return jax.jacfwd(d2_r_d_phi2_func)(phi_val)
    
    @jit
    def d_l_d_phi_func(phi_val):
        d_r_d_phi = d_r_d_phi_func(phi_val)
        return jnp.linalg.norm(d_r_d_phi, axis=0, keepdims=True)
    
    @jit
    def d2_l_d_phi2_func(phi_val):
        return jax.jacfwd(d_l_d_phi_func)(phi_val)
    
    @jit
    def manual_cross_product(a, b):
        nphi = a.shape[0]
        def body_fun(i, result):
            ax, ay, az = a[i]
            bx, by, bz = b[i]
            cp_x = ay * bz - az * by
            cp_y = az * bx - ax * bz
            cp_z = ax * by - ay * bx
            result = result.at[i].set(jnp.array([cp_x, cp_y, cp_z]))
            return result
        result = jnp.zeros_like(a)
        result = lax.fori_loop(0, nphi, body_fun, result)
        return result

    d_r_d_phi = jax.vmap(d_r_d_phi_func)(phi_vals)
    d2_r_d_phi2 = jax.vmap(d2_r_d_phi2_func)(phi_vals)
    d_l_d_phi = jax.vmap(d_l_d_phi_func)(phi_vals)
    d2_l_d_phi2 = jax.vmap(d2_l_d_phi2_func)(phi_vals)
    d3_r_d_phi3 = jax.vmap(d3_r_d_phi3_func)(phi_vals)

    tangent = d_r_d_phi / d_l_d_phi
    d_tangent_d_l = (-d_r_d_phi * d2_l_d_phi2 / d_l_d_phi + d2_r_d_phi2) / (d_l_d_phi * d_l_d_phi)
    curvature = jnp.linalg.norm(d_tangent_d_l, axis=1, keepdims=True)
    normal = d_tangent_d_l / curvature
    # binormal = jnp.cross(tangent, normal)
    binormal = manual_cross_product(tangent, normal)
    # torsion_numerator = jnp.sum(d_r_d_phi * jnp.cross(d2_r_d_phi2, d3_r_d_phi3), axis=1)
    # torsion_denominator = jnp.linalg.norm(jnp.cross(d_r_d_phi, d2_r_d_phi2), axis=1)**2
    torsion_numerator = jnp.sum(d_r_d_phi * manual_cross_product(d2_r_d_phi2, d3_r_d_phi3), axis=1)
    torsion_denominator = jnp.linalg.norm(manual_cross_product(d_r_d_phi, d2_r_d_phi2), axis=1)**2
    torsion = torsion_numerator / torsion_denominator

    B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
    G0 = sG * B0 / B0_over_abs_G0
    axis_length = 2 * jnp.pi / B0_over_abs_G0
    varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1, 0] + d_l_d_phi[1:, 0])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

    @jit
    def spectral_diff_matrix_jax():
        n= nphi
        xmin = 0
        xmax=2 * jnp.pi / nfp
        h = 2 * jnp.pi / n
        kk = jnp.arange(1, n)
        n_half = n // 2
        topc = 1 / jnp.sin(jnp.arange(1, n_half + 1) * h / 2)
        temp = jnp.concatenate((topc, jnp.flip(topc[:n_half])))
        col1 = jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))
        row1 = -col1
        vals = jnp.concatenate((row1[-1:0:-1], col1))
        a, b = jnp.ogrid[0:len(col1), len(row1)-1:-1:-1]
        return 2 * jnp.pi / (xmax - xmin) * vals[a + b]

    d_d_phi = spectral_diff_matrix_jax()
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    d_d_varphi = d_d_phi / d_varphi_d_phi

    # Determine helicity
    jax_normal_cartesian = normal.transpose()
    jax_normal_cylindrical = jnp.array( [jax_normal_cartesian[0]  * jnp.cos(phi_vals) + jax_normal_cartesian[1]  * jnp.sin(phi_vals), - jax_normal_cartesian[0]  * jnp.sin(phi_vals) + jax_normal_cartesian[1]  * jnp.cos(phi_vals), jax_normal_cartesian[2]])
    normal_cylindrical = jax_normal_cylindrical.transpose()
    x_positive = normal_cylindrical[:, 0] >= 0
    z_positive = normal_cylindrical[:, 2] >= 0
    quadrant = 1 * x_positive * z_positive \
             + 2 * ~x_positive * z_positive \
             + 3 * ~x_positive * ~z_positive \
             + 4 * x_positive * ~z_positive
    quadrant = jnp.append(quadrant, quadrant[0])
    delta_quadrant = quadrant[1:] - quadrant[:-1]
    increment = jnp.sum(1 * (quadrant[:-1] == 4) * (quadrant[1:] == 1))
    decrement = jnp.sum(1 * (quadrant[:-1] == 1) * (quadrant[1:] == 4))
    counter = jnp.sum(delta_quadrant) + increment - decrement
    helicity = counter * spsi * sG

    @jit
    def sigma_equation_residual():
        etaOcurv2 = etabar**2 / curvature[:,0]**2
        eq = jnp.matmul(d_d_varphi, sigma) \
        + (iota + helicity * nfp) * (etaOcurv2**2 + 1 + sigma**2) \
        - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0
        return eq

    res = sigma_equation_residual()

    X1c = etabar / curvature
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    p = + X1c * X1c + Y1s * Y1s + Y1c * Y1c
    q = - X1c * Y1s
    elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))

    if debug:
        return tangent.transpose(), normal.transpose(), binormal.transpose(), curvature[:,0], torsion, G0, axis_length, varphi, d_d_varphi, res, sigma, iota
    else:
        return res, elongation

@jit
def objective_function(params):
    sigma = jnp.array(params[0:nphi])
    rc = jnp.concatenate([jnp.array([1]),params[nphi:nphi+3]])
    zs = jnp.concatenate([jnp.array([0]),params[nphi+3:nphi+6]])
    etabar = params[-2]
    iota = params[-1]
    residuals, elongation = nacx_residual(eR=rc, eZ=zs, etabar=etabar, sigma=sigma, iota=iota)
    return jnp.sum(residuals**2)/nphi + jnp.sum(elongation**2)/nphi + (iota-iota_desired)**2

@jit
def grad_objective_function(params):
    grad_func = jax.grad(objective_function)(params)
    return grad_func

print('Do optimization')
zs=zs[1:]
rc=rc[1:]
sigma = jnp.zeros(nphi)
initial_params = jnp.concatenate([sigma,rc,zs,jnp.array([etabar,iota_desired])])
# print('Initial sum squares objective function:      {}'.format(jnp.sum(objective_function(initial_params))**2))
# print('Initial sum squares grad objective function: {}'.format(jnp.sum(grad_objective_function(initial_params))**2))

start_time = time();objective_function(initial_params);     print('Calculating JAX      values took {} seconds'.format(time() - start_time))
start_time = time();grad_objective_function(initial_params);print('Calculating JAX grad values took {} seconds'.format(time() - start_time))
start_time = time();objective_function(initial_params);     print('Calculating JAX      again  took {} seconds'.format(time() - start_time))
start_time = time();grad_objective_function(initial_params);print('Calculating JAX grad again  took {} seconds'.format(time() - start_time))
start_time = time();objective_function(initial_params);     print('Calculating JAX      again  took {} seconds'.format(time() - start_time))
start_time = time();grad_objective_function(initial_params);print('Calculating JAX grad again  took {} seconds'.format(time() - start_time))
exit()
from scipy.optimize import minimize, least_squares
import numpy as np
result = minimize(objective_function, initial_params, method='BFGS', jac=grad_objective_function, options={'disp': True})
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
stel.plot()
exit()

## DO DEBIGGING
start_time=time()
stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi)
print('Calculating pyQSC values took {} seconds'.format(time() - start_time))
start_time = time()
jax_tangent_cartesian, jax_normal_cartesian, jax_binormal_cartesian, jax_curvature, jax_torsion, jax_G0, jax_axis_length, jax_varphi, jax_d_d_varphi, res, sigma, iota = nacx_residual(rc, zs, nphi, debug=True)
intermediate_time = time();print('Calculating JAX values took {} seconds'.format(time() - start_time))

## Test that the JAX values are the same as the pyQSC values
a_tolerance = 1e-6
r_tolerance = 1e-5
phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
jax_tangent_cylindrical = jnp.array([jax_tangent_cartesian[0] * jnp.cos(phi) + jax_tangent_cartesian[1] * jnp.sin(phi), - jax_tangent_cartesian[0] * jnp.sin(phi) + jax_tangent_cartesian[1] * jnp.cos(phi), jax_tangent_cartesian[2]])
jax_normal_cylindrical = jnp.array( [jax_normal_cartesian[0]  * jnp.cos(phi) + jax_normal_cartesian[1]  * jnp.sin(phi), - jax_normal_cartesian[0]  * jnp.sin(phi) + jax_normal_cartesian[1]  * jnp.cos(phi), jax_normal_cartesian[2]])
jax_binormal_cylindrical = jnp.array([jax_binormal_cartesian[0] * jnp.cos(phi) + jax_binormal_cartesian[1] * jnp.sin(phi), - jax_binormal_cartesian[0] * jnp.sin(phi) + jax_binormal_cartesian[1] * jnp.cos(phi), jax_binormal_cartesian[2]])
assert jnp.allclose(jax_tangent_cylindrical, stel.tangent_cylindrical.transpose(), atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_normal_cylindrical, stel.normal_cylindrical.transpose(), atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_binormal_cylindrical, stel.binormal_cylindrical.transpose(), atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_curvature, stel.curvature, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_torsion, stel.torsion, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_G0, stel.G0, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_axis_length, stel.axis_length, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_varphi, stel.varphi, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_d_d_varphi, stel.d_d_varphi, atol=a_tolerance, rtol=r_tolerance)
from qsc.calculate_r1 import _residual, _jacobian
stel.sigma0 = sigma[0]
assert jnp.allclose(res, _residual(stel, jnp.concatenate([jnp.array([iota]), sigma[1:]])), atol=a_tolerance, rtol=r_tolerance)