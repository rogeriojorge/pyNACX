import jax
import jax.numpy as jnp
from jax import jit
from qsc import Qsc
from time import time
from functools import partial
import matplotlib.pyplot as plt

@partial(jit, static_argnums=(5,6,7,8,9,10,11,12))
def nacx_residual(eR=jnp.array([1, 0.1]), eZ=jnp.array([0, 0.1]), etabar=1.0, sigma=jnp.zeros(31), iota=0.4,
                  nphi=31, sigma0=0, I2=0, spsi=1, sG=1, B0=1, nfp=2, debug=False):
    assert nphi % 2 == 1, 'nphi must be odd'
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

    def sigma_equation_jacobian(curvature, sigma, etabar, d_d_varphi, iota, helicity):
        etaOcurv2 = etabar**2 / curvature**2
        jac = d_d_varphi + (iota + helicity * nfp) * 2 * jnp.diag(sigma)
        return jac.at[:, 0].set(etaOcurv2**2 + 1 + sigma**2)

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

    sigma = sigma.at[0].set(sigma0)
    res = sigma_equation_residual(curvature, torsion, sigma, etabar, d_d_varphi, iota, G0, B0, helicity)
    jac = sigma_equation_jacobian(curvature, sigma, etabar, d_d_varphi, iota, helicity)

    X1c = etabar / curvature
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    p = + X1c * X1c + Y1s * Y1s + Y1c * Y1c
    q = - X1c * Y1s
    elongation = (p + jnp.sqrt(p * p - 4 * q * q)) / (2 * jnp.abs(q))

    if debug:
        return tangent_cylindrical, normal_cylindrical, binormal_cylindrical, curvature, torsion, G0, axis_length, varphi, d_d_varphi, res, sigma, iota, helicity, elongation, jac
    else:
        return res, elongation
