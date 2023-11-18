import jax
import jax.numpy as jnp
from jax import jit
from qsc import Qsc
from time import time
from functools import partial
import matplotlib.pyplot as plt

nfp = 2
rc = jnp.array([1, 0.1, 0.01, 0.001])
zs = jnp.array([0, 0.1, 0.01, 0.001])
sG = 1
spsi = 1
I2 = 0
B0 = 1
etabar = 0.9
sigma0=0
iota_desired = 0.4
nphi = 31
assert nphi % 2 == 1, 'nphi must be odd'

@partial(jit, static_argnums=(5,6,7))
def nacx_residual(eR, eZ, etabar=1.0, sigma=jnp.zeros(nphi)+0.01, iota=iota_desired, nphi=nphi, sigma0=sigma0, debug=False):
    phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    d_phi = phi[1] - phi[0]
    nfourier = max(len(eR), len(eZ))
    n_values = jnp.arange(nfourier) * nfp

    @jit
    def compute_terms(jn):
        n = n_values[jn]
        sinangle = jnp.sin(n * phi)
        cosangle = jnp.cos(n * phi)
        return jnp.array([eR[jn] * cosangle, eZ[jn] * sinangle,
            eR[jn] * (-n * sinangle), eZ[jn] * (n * cosangle),
            eR[jn] * (-n * n * cosangle), eZ[jn] * (-n * n * sinangle),
            eR[jn] * (n * n * n * sinangle), eZ[jn] * (-n * n * n * cosangle)])

    def spectral_diff_matrix_jax(n, xmin=0, xmax=2*jnp.pi):
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

    def sigma_equation_residual(curvature, torsion, sigma, etabar, d_d_varphi, iota, G0, B0, helicity):
        etaOcurv2 = etabar**2 / curvature**2
        return jnp.matmul(d_d_varphi, sigma) \
        + (iota + helicity * nfp) * (etaOcurv2**2 + 1 + sigma**2) \
        - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0

    def determine_helicity(normal_cylindrical):
        x_positive = normal_cylindrical[:, 0] >= 0
        z_positive = normal_cylindrical[:, 2] >= 0
        quadrant = 1 * x_positive * z_positive + 2 * (~x_positive) * z_positive \
                 + 3 * (~x_positive) * (~z_positive) + 4 * x_positive * (~z_positive)
        quadrant = jnp.append(quadrant, quadrant[0])
        delta_quadrant = quadrant[1:] - quadrant[:-1]
        increment = jnp.sum((quadrant[:-1] == 4) & (quadrant[1:] == 1))
        decrement = jnp.sum((quadrant[:-1] == 1) & (quadrant[1:] == 4))
        return (jnp.sum(delta_quadrant) + increment - decrement) * spsi * sG

    summed_values = jnp.sum(jax.vmap(compute_terms)(jnp.arange(nfourier)), axis=0)

    R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp = summed_values
    d_l_d_phi = jnp.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    B0_over_abs_G0 = nphi / jnp.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    d_l_d_varphi = abs_G0_over_B0
    G0 = sG * abs_G0_over_B0 * B0

    d_r_d_phi_cylindrical = jnp.stack([R0p, R0, Z0p]).T
    d2_r_d_phi2_cylindrical = jnp.stack([R0pp - R0, 2 * R0p, Z0pp]).T
    d3_r_d_phi3_cylindrical = jnp.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).T

    tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, None]
    d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, None] * d_l_d_phi[:, None])

    curvature = jnp.sqrt(jnp.sum(d_tangent_d_l_cylindrical**2, axis=1))
    axis_length = jnp.sum(d_l_d_phi) * d_phi * nfp
    varphi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:])]) * (0.5 * d_phi * 2 * jnp.pi / axis_length)

    normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, None]
    binormal_cylindrical = jnp.cross(tangent_cylindrical, normal_cylindrical)

    torsion_numerator = jnp.sum(d_r_d_phi_cylindrical * jnp.cross(d2_r_d_phi2_cylindrical, d3_r_d_phi3_cylindrical), axis=1)
    torsion_denominator = jnp.sum(jnp.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical)**2, axis=1)
    torsion = torsion_numerator / torsion_denominator

    d_d_phi = spectral_diff_matrix_jax(nphi, xmax=2 * jnp.pi / nfp)
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
    helicity = determine_helicity(normal_cylindrical)

    res = sigma_equation_residual(curvature, torsion, sigma, etabar, d_d_varphi, iota, G0, B0, helicity)

    X1c = etabar / curvature
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    p = + X1c * X1c + Y1s * Y1s + Y1c * Y1c
    q = - X1c * Y1s
    elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))

    if debug:
        return tangent_cylindrical, normal_cylindrical, binormal_cylindrical, curvature, torsion, G0, axis_length, varphi, d_d_varphi, res, sigma, iota, helicity, elongation
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
    return 1e3*jnp.sum(residuals**2)/nphi + 1e-3*jnp.sum(elongation**2)/nphi + 1e3*(iota-iota_desired)**2
    # return jnp.array(residuals)

# print('Do optimization')
# zs=zs[1:]
# rc=rc[1:]
# sigma = jnp.zeros(nphi)
# initial_params = jnp.concatenate([sigma,rc,zs,jnp.array([etabar,iota_desired])])
# Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
# print('Initial objective function: {}'.format(objective_function(initial_params)))

# from scipy.optimize import minimize, least_squares
# import numpy as np
# result = minimize(objective_function, initial_params, method='BFGS', jac=jax.grad(objective_function), options={'disp': True})
# optimized_params = result.x

# from jax.scipy.optimize import minimize
# result = minimize(objective_function, initial_params, method="BFGS")
# optimized_params = result.x

# import jaxopt
# tol_optimization=1e-6
# max_nfev_optimization=10000
# optimizer = jaxopt.ScipyMinimize(fun=objective_function, method='L-BFGS-B', tol=tol_optimization, maxiter=max_nfev_optimization, jit=True)#, options={'jac':True})
# optimized_params, state = optimizer.run(initial_params)

# optimized_sigma, optimized_rc, optimized_zs, optimized_etabar, optimized_iota = optimized_params[0:nphi], optimized_params[nphi:nphi+3], optimized_params[nphi+3:nphi+6], optimized_params[-2], optimized_params[-1]
# print('Optimized rc: {}'.format(optimized_rc))
# print('Optimized zs: {}'.format(optimized_zs))
# print('Optimized etabar: {}'.format(optimized_etabar))
# print('Optimized iota: {}'.format(optimized_iota))
# print('Optimized objective function: {}'.format(objective_function(optimized_params)))
# objective_function(optimized_params)
# objective_function(optimized_params)
# rc = jnp.concatenate([jnp.array([1]), optimized_rc])
# zs = jnp.concatenate([jnp.array([0]), optimized_zs])
# etabar = optimized_etabar
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar)
# print(f'True iota: {stel.iota}')
# stel.plot()
# stel.plot_boundary(r=0.1)
# exit()

## DO DEBIGGING
# start_time=time()
stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi)
# print('Calculating pyQSC values took {} seconds'.format(time() - start_time))
# start_time = time()
jax_tangent_cylindrical, jax_normal_cylindrical, jax_binormal_cylindrical, jax_curvature, jax_torsion, jax_G0, jax_axis_length, jax_varphi, jax_d_d_varphi, res, sigma, iota, helicity, elongation = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, debug=True)
# intermediate_time = time();print('Calculating JAX values took {} seconds'.format(time() - start_time))
# jax_tangent_cylindrical, jax_normal_cylindrical, jax_binormal_cylindrical, jax_curvature, jax_torsion, jax_G0, jax_axis_length, jax_varphi, jax_d_d_varphi, res, sigma, iota, helicity, elongation = nacx_residual(rc, zs, nphi, debug=True)
# final_time = time();print('Calculating JAX again took {} seconds'.format(final_time - intermediate_time))

## Test that the JAX values are the same as the pyQSC values
a_tolerance = 1e-6
r_tolerance = 1e-5
assert jnp.allclose(jax_tangent_cylindrical, stel.tangent_cylindrical, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_normal_cylindrical, stel.normal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_binormal_cylindrical, stel.binormal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_curvature, stel.curvature, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_torsion, stel.torsion, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_G0, stel.G0, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_axis_length, stel.axis_length, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_varphi, stel.varphi, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jax_d_d_varphi, stel.d_d_varphi, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(helicity, stel.helicity, atol=a_tolerance, rtol=r_tolerance)
from qsc.calculate_r1 import _residual, _jacobian
stel.sigma0 = sigma[0]
assert jnp.allclose(res, _residual(stel, jnp.concatenate([jnp.array([iota]), sigma[1:]])), atol=a_tolerance, rtol=r_tolerance)