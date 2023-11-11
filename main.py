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

import jax
import jax.numpy as jnp

nphi = 21
nfp = 2
rc = jnp.array([1, 0.1])
zs = jnp.array([0, 0.1])
phi = jnp.linspace(0, 2 * jnp.pi / nfp, nphi, endpoint=False)

def pos_vector_component(eR, eZ, component_index, phi_val):
    R = jnp.sum(eR * jnp.cos(jnp.arange(len(eR)) * phi_val * nfp))
    Z = jnp.sum(eZ * jnp.cos(jnp.arange(len(eZ)) * phi_val * nfp))
    if component_index == 0:
        return R * jnp.cos(phi_val)
    elif component_index == 1:
        return R * jnp.sin(phi_val)
    else:
        return Z

def frenet_frame(eR, eZ):
    tangent = []
    for i in range(3):
        component_fn = lambda phi_val: pos_vector_component(eR, eZ, i, phi_val)
        d_component_d_phi = jax.vmap(jax.grad(component_fn))(phi)
        tangent.append(d_component_d_phi)
    
    return jnp.array(tangent)

from qsc import Qsc
from matplotlib import pyplot as plt
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi)
stel_tangent = stel.tangent_cylindrical.transpose()
jax_tangent = frenet_frame(rc, zs)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i in range(3):
    axes[i].plot(stel_tangent[i], label='Stel Tangent Component {}'.format(i+1))
    axes[i].plot(jax_tangent[i], label='JAX Tangent Component {}'.format(i+1))
    axes[i].legend()
    axes[i].set_ylabel('Component {}'.format(i+1))
plt.show()