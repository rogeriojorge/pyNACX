import tensorflow as tf
import numpy as np
from scipy.linalg import solve
from functools import partial
tf.config.optimizer.set_jit(True)

@tf.function
def nacx_residual(eR=tf.constant([1, 0.1]), eZ=tf.constant([0, 0.1]), etabar=1.0,
                  nphi=31, sigma0=0.0, I2=0.0, spsi=1.0, sG=1.0, B0=1.0, nfp=2, debug=False):
    assert nphi % 2 == 1, 'nphi must be odd'
    phi = tf.linspace(0., 2 * np.pi / nfp * (1-1/nphi), nphi)
    d_phi = phi[1] - phi[0]
    nfourier = max(len(eR), len(eZ))
    n_values = tf.cast(tf.range(nfourier) * nfp, tf.float32)
    n_phi = tf.cast(n_values[:, None], tf.float32) * phi[None, :]
    sinangles = tf.sin(n_phi)
    cosangles = tf.cos(n_phi)
    eR_exp = tf.reshape(eR, (-1, 1))
    eZ_exp = tf.reshape(eZ, (-1, 1))
    summed_values = tf.stack([
        tf.reduce_sum(eR_exp * cosangles, axis=0), 
        tf.reduce_sum(eZ_exp * sinangles, axis=0),
        tf.reduce_sum(-eR_exp * n_values[:, None] * sinangles, axis=0),
        tf.reduce_sum(eZ_exp * n_values[:, None] * cosangles, axis=0),
        tf.reduce_sum(-eR_exp * tf.square(n_values[:, None]) * cosangles, axis=0),
        tf.reduce_sum(-eZ_exp * tf.square(n_values[:, None]) * sinangles, axis=0),
        tf.reduce_sum(eR_exp * tf.pow(n_values[:, None], 3) * sinangles, axis=0),
        tf.reduce_sum(-eZ_exp * tf.pow(n_values[:, None], 3) * cosangles, axis=0)
    ], axis=0)
    R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp = tf.unstack(summed_values, axis=0)

    d_l_d_phi = tf.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    B0_over_abs_G0 = nphi / tf.reduce_sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    d_l_d_varphi = abs_G0_over_B0
    G0 = sG * abs_G0_over_B0 * B0

    d_r_d_phi_cylindrical = tf.stack([R0p, R0, Z0p], axis=1)
    d2_r_d_phi2_cylindrical = tf.stack([R0pp - R0, 2 * R0p, Z0pp], axis=1)
    d3_r_d_phi3_cylindrical = tf.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp], axis=1)

    tangent_cylindrical = d_r_d_phi_cylindrical / tf.expand_dims(d_l_d_phi, axis=-1)
    d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * tf.expand_dims(d2_l_d_phi2, axis=-1) / tf.expand_dims(d_l_d_phi, axis=-1) 
                                 + d2_r_d_phi2_cylindrical) / tf.square(tf.expand_dims(d_l_d_phi, axis=-1))

    curvature = tf.sqrt(tf.reduce_sum(tf.square(d_tangent_d_l_cylindrical), axis=1))
    axis_length = tf.reduce_sum(d_l_d_phi) * d_phi * nfp
    varphi = tf.concat([[0.], tf.cumsum((d_l_d_phi[:-1] + d_l_d_phi[1:]) * 0.5 * d_phi * 2 * np.pi / axis_length)], axis=0)

    normal_cylindrical = d_tangent_d_l_cylindrical / tf.expand_dims(curvature, axis=-1)
    binormal_cylindrical = tf.linalg.cross(tangent_cylindrical, normal_cylindrical)

    torsion_numerator = tf.reduce_sum(d_r_d_phi_cylindrical * tf.linalg.cross(d2_r_d_phi2_cylindrical, d3_r_d_phi3_cylindrical), axis=1)
    torsion_denominator = tf.reduce_sum(tf.square(tf.linalg.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical)), axis=1)
    torsion = torsion_numerator / torsion_denominator

    def spectral_diff_matrix_tf():
        n = nphi
        xmin = 0.0
        xmax = 2 * np.pi / nfp
        h = 2 * np.pi / n
        kk = tf.cast(tf.range(1, n), tf.float32)
        n_half = n // 2
        topc = 1 / tf.sin(tf.range(1, n_half + 1, dtype=tf.float32) * h / 2)
        temp = tf.concat([topc, tf.reverse(topc[:n_half], axis=[0])], axis=0)
        col1 = tf.concat([tf.constant([0.0]), 0.5 * ((-1) ** kk) * temp], axis=0)
        row1 = -col1
        vals = tf.concat([row1[-1:0:-1], col1], axis=0)
        a, b = tf.meshgrid(tf.range(len(col1)), len(row1) - 1 - tf.range(len(row1)), indexing='ij')
        return 2 * np.pi / (xmax - xmin) * tf.gather(vals, a + b, axis=0)

    def determine_helicity_tf(normal_cylindrical):
        x_positive = normal_cylindrical[:, 0] >= 0
        z_positive = normal_cylindrical[:, 2] >= 0
        quadrant = 1 * tf.cast(x_positive & z_positive, tf.float32) + \
                2 * tf.cast(~x_positive & z_positive, tf.float32) + \
                3 * tf.cast(~x_positive & ~z_positive, tf.float32) + \
                4 * tf.cast(x_positive & ~z_positive, tf.float32)
        quadrant = tf.concat([quadrant, [quadrant[0]]], axis=0)
        delta_quadrant = quadrant[1:] - quadrant[:-1]
        increment = tf.reduce_sum(tf.cast((quadrant[:-1] == 4) & (quadrant[1:] == 1), tf.float32))
        decrement = tf.reduce_sum(tf.cast((quadrant[:-1] == 1) & (quadrant[1:] == 4), tf.float32))
        return (tf.reduce_sum(delta_quadrant) + increment - decrement) * spsi * sG
    
    d_d_phi = spectral_diff_matrix_tf()
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    d_d_varphi = d_d_phi / d_varphi_d_phi[:, None]
    helicity = determine_helicity_tf(normal_cylindrical)

    def replace_first_element_tf(x, new_value):
        return tf.concat([[new_value], x[1:]], axis=0)

    def sigma_equation_residual_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp, spsi, torsion, I2, B0, G0):
        iota = x[0]
        sigma = replace_first_element_tf(x, sigma0)
        etaOcurv2 = etabar**2 / tf.square(curvature)
        residual = tf.linalg.matvec(d_d_varphi, sigma) \
            + (iota + helicity * nfp) * (tf.square(etaOcurv2) + 1 + tf.square(sigma)) \
            - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0
        return residual

    def sigma_equation_jacobian_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp):
        iota = x[0]
        sigma = replace_first_element_tf(x, sigma0)
        etaOcurv2 = etabar**2 / tf.square(curvature)
        diagonal = tf.linalg.diag(sigma)
        jac = d_d_varphi + (iota + helicity * nfp) * 2 * diagonal
        jac_col_updated = tf.tensor_scatter_nd_update(jac, [[i, 0] for i in range(jac.shape[0])], etaOcurv2**2 + 1 + sigma**2)
        return jac_col_updated

    def newton_tf(x0, niter, d_d_varphi, sigma0, etabar, curvature, helicity, nfp, spsi, torsion, I2, B0, G0):
        x = x0
        for _ in range(niter):
            residual = sigma_equation_residual_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp, spsi, torsion, I2, B0, G0)
            jacobian = sigma_equation_jacobian_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp)
            step = tf.linalg.solve(jacobian, tf.expand_dims(-residual, axis=-1))
            x += tf.squeeze(step)
        return x
    
    x0 = tf.fill([nphi], sigma0)
    x0 = replace_first_element_tf(x0, 0.0)
    sigma = newton_tf(x0, 4, d_d_varphi, sigma0, etabar, curvature, helicity, nfp, spsi, torsion, I2, B0, G0)
    iota = sigma[0]
    iotaN = iota + helicity * nfp
    sigma = replace_first_element_tf(sigma, sigma0)
    x = tf.concat([[iota], sigma[1:]], axis=0)

    res = sigma_equation_residual_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp, spsi, torsion, I2, B0, G0)
    jac = sigma_equation_jacobian_tf(x, d_d_varphi, sigma0, etabar, curvature, helicity, nfp)

    # Perform additional calculations
    X1c = etabar / curvature
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    p = X1c * X1c + Y1s * Y1s + Y1c * Y1c
    q = -X1c * Y1s
    elongation = (p + tf.sqrt(p * p - 4 * q * q)) / (2 * tf.abs(q))

    # Calculate derivatives
    d_X1c_d_varphi = tf.matmul(d_d_varphi, tf.reshape(X1c, [-1, 1]))
    d_Y1s_d_varphi = tf.matmul(d_d_varphi, tf.reshape(Y1s, [-1, 1]))
    d_Y1c_d_varphi = tf.matmul(d_d_varphi, tf.reshape(Y1c, [-1, 1]))

    # Additional calculations
    factor = spsi * B0 / d_l_d_varphi
    tn = sG * B0 * curvature
    nt = tn
    bb = factor * (X1c * tf.squeeze(d_Y1s_d_varphi) - iotaN * X1c * Y1c)
    nn = factor * (tf.squeeze(d_X1c_d_varphi) * Y1s + iotaN * X1c * Y1c)
    bn = factor * (-sG * spsi * d_l_d_varphi * torsion - iotaN * X1c * X1c)
    nb = factor * (tf.squeeze(d_Y1c_d_varphi) * Y1s - tf.squeeze(d_Y1s_d_varphi) * Y1c + sG * spsi * d_l_d_varphi * torsion + iotaN * (Y1s * Y1s + Y1c * Y1c))

    # Final calculations
    grad_B_colon_grad_B = tn * tn + nt * nt + bb * bb + nn * nn + nb * nb + bn * bn
    L_grad_B = B0 * tf.sqrt(2 / grad_B_colon_grad_B)
    inv_L_grad_B = 1.0 / L_grad_B

    return (tangent_cylindrical, normal_cylindrical, binormal_cylindrical, curvature, torsion, G0, axis_length, varphi, d_d_varphi, res, sigma, iota, helicity, elongation, jac, inv_L_grad_B, phi, d_d_phi) if debug else (iota, elongation, inv_L_grad_B)

