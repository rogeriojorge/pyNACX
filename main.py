## Solve near-axis equations using JAX

# from scipy.optimize import root
# import matplotlib.pyplot as plt

# def newton(f, x_0, tol=1e-5):
#     f_prime = jax.grad(f)
#     def q(x):
#         return x - f(x) / f_prime(x)

#     error = tol + 1
#     x = x_0
#     while error > tol:
#         y = q(x)
#         error = abs(x - y)
#         x = y
        
#     return x

# f = lambda x: jnp.sin(4 * (x - 1/4)) + x + x**20 - 1
# x = jnp.linspace(0, 1, 100)

# fig, ax = plt.subplots()
# ax.plot(x, f(x), label='$f(x)$')
# ax.axhline(ls='--', c='k')
# ax.set_xlabel('$x$', fontsize=12)
# ax.set_ylabel('$f(x)$', fontsize=12)
# ax.legend(fontsize=12)
# plt.show()

# import jax
# import jax.numpy as jnp
# nphi = 21
# nfp = 2
# rc = jnp.array([1, 0.1])
# zs = jnp.array([0, 0.1])
# phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
# def pos_vector(eR, eZ):
#     R = jnp.sum(eR[:, None] * jnp.cos(jnp.arange(len(eR))[:, None] * phi * nfp), axis=0)
#     Z = jnp.sum(eZ[:, None] * jnp.cos(jnp.arange(len(eZ))[:, None] * phi * nfp), axis=0)
#     return jnp.array([R*jnp.cos(phi), R*jnp.sin(phi), Z])
# def frenet_frame(eR, eZ):
#     pos = pos_vector(eR, eZ)
#     jacobian_pos = jax.jacfwd(pos_vector)(eR, eZ)
#     tangent = jacobian_pos[0]
#     tangent /= jnp.linalg.norm(tangent, axis=0)
#     d_tangent_d_l = jax.jacfwd(jax.jacfwd(pos_vector))(eR, eZ)
#     normal = d_tangent_d_l[0, 0] / jnp.linalg.norm(d_tangent_d_l[0, 0], axis=0)
#     tangent = tangent[:, :normal.shape[1]]
#     binormal = jnp.cross(tangent, normal)
#     return tangent, normal, binormal

# def curvature(eR, eZ):
#     _, _, binormal = frenet_frame(eR, eZ)
#     d_tangent_d_l = jax.jacfwd(jax.jacfwd(pos_vector))(eR, eZ)
#     d_tangent_d_l /= jnp.linalg.norm(d_tangent_d_l, axis=0)
#     return jnp.linalg.norm(d_tangent_d_l, axis=0), binormal

# def torsion(eR, eZ):
#     tangent, _, binormal = frenet_frame(eR, eZ)
#     d_tangent_d_l = jax.jacfwd(jax.jacfwd(pos_vector))(eR, eZ)
#     d_tangent_d_l /= jnp.linalg.norm(d_tangent_d_l, axis=0)
#     d_binormal_d_l = jax.jacfwd(jax.jacfwd(jnp.cross))(tangent, binormal)
#     return jnp.sum(d_binormal_d_l * binormal, axis=0)

# def axis_length(eR, eZ):
#     tangent, _, _ = frenet_frame(eR, eZ)
#     d_l_d_phi = jnp.linalg.norm(tangent, axis=0)
#     return jnp.sum(d_l_d_phi) * (2 * jnp.pi / nfp) / nphi

# def axis_helicity(eR, eZ):
#     tangent, _, _ = frenet_frame(eR, eZ)
#     d_l_d_phi = jnp.linalg.norm(tangent, axis=0)
#     d_phi = 2 * jnp.pi / nphi
#     d2_l_d_phi2 = (d_l_d_phi[2:] - d_l_d_phi[:-2]) / (2 * d_phi)
#     return jnp.sum(d2_l_d_phi2) * d_phi * nfp

# from qsc import Qsc
# from matplotlib import pyplot as plt
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi)
# plt.plot(curvature(rc,zs))
# plt.plot(stel.curvature)
# plt.show()

# import jax
# import jax.numpy as jnp
# from qsc import Qsc
# from matplotlib import pyplot as plt
# from time import time

# nphi = 101
# nfp = 2
# rc = jnp.array([1, 0.1])
# zs = jnp.array([0, 0.1])
# phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi)

# print('Defining Functions')
# start = time()

# def pos_vector_component(eR, eZ, phi_val):
#     phi_val_expanded = jnp.atleast_1d(phi_val)
#     rc_harmonics = jnp.arange(rc.size)
#     zs_harmonics = jnp.arange(zs.size)
#     rc_cosines = jnp.cos(rc_harmonics * phi_val_expanded * nfp)
#     zs_sines   = jnp.sin(zs_harmonics * phi_val_expanded * nfp)
#     R = jnp.sum(rc * rc_cosines, axis=0)
#     Z = jnp.sum(zs * zs_sines, axis=0)
#     return jnp.array([R * jnp.cos(phi_val), R * jnp.sin(phi_val), Z])

# def d_r_d_phi(eR, eZ, phi_val):
#     result = []
#     for i in range(3):
#         component_fn = lambda p, i=i: pos_vector_component(eR, eZ, p)[i]
#         d_component_d_phi = jax.grad(component_fn)(phi_val)
#         result.append(d_component_d_phi)
#     return jnp.array(result)

# def d2_r_d_phi2(eR, eZ, phi_val):
#     result = []
#     for i in range(3):
#         d_r_d_phi_fn = lambda p, i=i: d_r_d_phi(eR, eZ, p)[i]
#         d2_component_d_phi2 = jax.grad(d_r_d_phi_fn)(phi_val)
#         result.append(d2_component_d_phi2)
#     return jnp.array(result)

# def d3_r_d_phi3(eR, eZ, phi_val):
#     result = []
#     for i in range(3):
#         d2_r_d_phi2_fn = lambda p, i=i: d2_r_d_phi2(eR, eZ, p)[i]
#         d3_component_d_phi3 = jax.grad(d2_r_d_phi2_fn)(phi_val)
#         result.append(d3_component_d_phi3)
#     return jnp.array(result)

# def tangent_vector(eR, eZ, phi_val):
#     d_r_d_phi_array = d_r_d_phi(eR, eZ, phi_val)
#     result = jnp.array(d_r_d_phi_array) / jnp.linalg.norm(jnp.array(d_r_d_phi_array), axis=0)
#     return result.transpose()

# def d_tangent_d_phi(eR, eZ, phi_val):
#     result = []
#     for i in range(3):
#         tangent_fn = lambda p, i=i: tangent_vector(eR, eZ, p)[i]
#         d_component_d_phi = jax.grad(tangent_fn)(phi_val)
#         result.append(d_component_d_phi)
#     return jnp.array(result)

# def d_l_d_phi(eR, eZ, phi_val):
#     d_r_d_phi_array = d_r_d_phi(eR, eZ, phi_val)
#     return jnp.linalg.norm(jnp.array(d_r_d_phi_array), axis=0)

# def curvature(eR, eZ, phi_val):
#     d_tangent_d_phi_array = d_tangent_d_phi(eR, eZ, phi_val)
#     d_l_d_phi_array = d_l_d_phi(eR, eZ, phi_val)
#     return jnp.linalg.norm(d_tangent_d_phi_array / d_l_d_phi_array, axis=0)

# def normal_vector(eR, eZ, phi_val):
#     d_tangent_d_phi_array = d_tangent_d_phi(eR, eZ, phi_val)
#     d_l_d_phi_array = d_l_d_phi(eR, eZ, phi_val)
#     curvature_array = curvature(eR, eZ, phi_val)
#     return d_tangent_d_phi_array / d_l_d_phi_array / curvature_array

# def binormal_vector(eR, eZ, phi_val):
#     tangent_array = tangent_vector(eR, eZ, phi_val)
#     normal_array = normal_vector(eR, eZ, phi_val)
#     return jnp.cross(tangent_array, normal_array)

# def d_normal_d_phi(eR, eZ, phi_val):
#     result = []
#     for i in range(3):
#         normal_fn = lambda p, i=i: normal_vector(eR, eZ, p)[i]
#         d_component_d_phi = jax.grad(normal_fn)(phi_val)
#         result.append(d_component_d_phi)
#     return jnp.array(result)

# def torsion(eR, eZ, phi_val):
#     d_l_d_phi_array = d_l_d_phi(eR, eZ, phi_val)
#     d_normal_d_phi_array = d_normal_d_phi(eR, eZ, phi_val)
#     binormal_array = binormal_vector(eR, eZ, phi_val)
#     return jnp.dot(d_normal_d_phi_array, binormal_array)/d_l_d_phi_array

# intermediate_time = time();print('  Defining Functions took {} seconds'.format(intermediate_time - start))
# print('Calculating tangent')
# jax_tangent_cartesian = jax.vmap(lambda p: tangent_vector(rc, zs, p))(phi).transpose()
# new_time = time();print('  Calculating tangent took {} seconds'.format(time() - intermediate_time))
# jax_normal_cartesian = jax.vmap(lambda p: normal_vector(rc, zs, p))(phi).transpose()
# new_time2 = time();print('  Calculating normal took {} seconds'.format(time() - new_time))
# jax_binormal_cartesian = jax.vmap(lambda p: binormal_vector(rc, zs, p))(phi).transpose()
# new_time3 = time();print('  Calculating binormal took {} seconds'.format(time() - new_time2))
# jax_curvature = jax.vmap(lambda p: curvature(rc, zs, p))(phi).transpose()
# new_time4 = time();print('  Calculating curvature took {} seconds'.format(time() - new_time3))
# jax_torsion = jax.vmap(lambda p: torsion(rc, zs, p))(phi).transpose()
# new_time5 = time();print('  Calculating torsion took {} seconds'.format(time() - new_time4))

import jax
import jax.numpy as jnp
from qsc import Qsc
from time import time
import matplotlib.pyplot as plt

nphi = 31
nfp = 2
rc = jnp.array([1, 0.1])
zs = jnp.array([0, 0.1])
phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi)

def compute_frenet_frames_and_curvature_torsion(eR, eZ, phi_vals):
    def pos_vector_component(phi_val):
        rc_cosines = jnp.cos(jnp.arange(eR.size) * phi_val * nfp)
        zs_sines = jnp.sin(jnp.arange(eZ.size) * phi_val * nfp)
        R = jnp.sum(eR * rc_cosines)
        Z = jnp.sum(eZ * zs_sines)
        return jnp.array([R * jnp.cos(phi_val), R * jnp.sin(phi_val), Z])

    def derivative_component(phi_val):
        return jax.jacfwd(pos_vector_component)(phi_val)

    def second_derivative_component(phi_val):
        return jax.jacfwd(derivative_component)(phi_val)

    d_r_d_phi = jax.vmap(derivative_component)(phi_vals)
    d2_r_d_phi2 = jax.vmap(second_derivative_component)(phi_vals)

    tangent = d_r_d_phi / jnp.linalg.norm(d_r_d_phi, axis=1, keepdims=True)
    curvature = jnp.linalg.norm(d2_r_d_phi2, axis=1) / jnp.linalg.norm(d_r_d_phi, axis=1)**3

    normal = d2_r_d_phi2 / jnp.linalg.norm(d2_r_d_phi2, axis=1, keepdims=True)
    binormal = jnp.cross(tangent, normal)

    def third_derivative_component(phi_val):
        return jax.jacfwd(lambda p: jnp.cross(derivative_component(p), normal_vector(p)))(phi_val)

    def normal_vector(phi_val):
        return second_derivative_component(phi_val) / jnp.linalg.norm(second_derivative_component(phi_val), axis=0)

    torsion = jax.vmap(third_derivative_component)(phi_vals)
    torsion = jnp.sum(torsion * binormal, axis=1)

    return tangent.transpose(), normal.transpose(), binormal.transpose(), curvature, torsion

start = time()
jax_tangent_cartesian, jax_normal_cartesian, jax_binormal_cartesian, jax_curvature, jax_torsion = compute_frenet_frames_and_curvature_torsion(rc, zs, phi)
intermediate_time = time();print('Calculating JAX values took {} seconds'.format(time() - start))

stel_tangent_cylindrical = stel.tangent_cylindrical.transpose()
stel_normal_cylindrical = stel.normal_cylindrical.transpose()
stel_binormal_cylindrical = stel.binormal_cylindrical.transpose()
print('Calculating pyQSC values took {} seconds'.format(time() - intermediate_time))

jax_tangent_cylindrical = jnp.array([jax_tangent_cartesian[0] * jnp.cos(phi) + jax_tangent_cartesian[1] * jnp.sin(phi), - jax_tangent_cartesian[0] * jnp.sin(phi) + jax_tangent_cartesian[1] * jnp.cos(phi), jax_tangent_cartesian[2]])
jax_normal_cylindrical = jnp.array( [jax_normal_cartesian[0]  * jnp.cos(phi) + jax_normal_cartesian[1]  * jnp.sin(phi), - jax_normal_cartesian[0]  * jnp.sin(phi) + jax_normal_cartesian[1]  * jnp.cos(phi), jax_normal_cartesian[2]])
jax_binormal_cylindrical = jnp.array([jax_binormal_cartesian[0] * jnp.cos(phi) + jax_binormal_cartesian[1] * jnp.sin(phi), - jax_binormal_cartesian[0] * jnp.sin(phi) + jax_binormal_cartesian[1] * jnp.cos(phi), jax_binormal_cartesian[2]])

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i in range(3):
    axes[0, 0].plot(stel_tangent_cylindrical[i], '--', label='Stel Tangent Component {}'.format(i+1))
    axes[0, 0].plot(jax_tangent_cylindrical[i], '.', label='JAX Tangent Component {}'.format(i+1))
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('Components of the Tangent Vector')
for i in range(3):
    axes[1, 0].plot(stel_normal_cylindrical[i], '--', label='Stel Normal Component {}'.format(i+1))
    axes[1, 0].plot(jax_normal_cylindrical[i], '.', label='JAX Normal Component {}'.format(i+1))
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('Components of the Normal Vector')
for i in range(3):
    axes[0, 1].plot(stel_binormal_cylindrical[i], '--', label='Stel Binormal Component {}'.format(i+1))
    axes[0, 1].plot(jax_binormal_cylindrical[i], '.', label='JAX Binormal Component {}'.format(i+1))
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Components of the Binormal Vector')
axes[1, 1].plot(stel.curvature, '--', label='Stel Curvature')
axes[1, 1].plot(jax_curvature, '.', label='JAX Curvature')
axes[1, 1].legend()
axes[1, 1].plot(stel.torsion, '--', label='Stel Torsion')
axes[1, 1].plot(jax_torsion, '.', label='JAX Torsion')
axes[1, 1].legend()
axes[1, 1].set_ylabel('Curvature, Torsion')
plt.show()