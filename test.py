from time import time
from qsc import Qsc
from main import nacx_residual
import jax.numpy as jnp

nfp = 2
rc = jnp.array([1, 0.1, 0.01, 0.001])
zs = jnp.array([0, 0.1, 0.01, 0.001])
sG = 1
spsi = 1
I2 = 0
B0 = 1
etabar = 0.9
sigma0=0.2
nphi = 71

start_time=time();stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC values took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc=rc, zs=zs, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=False)
print('Calculating JAX values took {} seconds'.format(time() - intermediate_time))
jax_tangent_cylindrical, jax_normal_cylindrical, jax_binormal_cylindrical, \
    jax_curvature, jax_torsion, jax_G0, jax_axis_length, jax_varphi, \
        jax_d_d_varphi, res, sigma, iota, helicity, elongation, jac, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0,
                                                                                    I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=True)
print('Calculating JAX again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=False)
print('Calculating JAX again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=True)
print('Calculating JAX again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=False)
print('Calculating JAX again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp, debug=True)
print('Calculating JAX again  took {} seconds'.format(time() - intermediate_time))

## Test that the JAX values are the same as the pyQSC values
a_tolerance = 1e-5
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
import matplotlib.pyplot as plt
# plt.plot(sigma)
# plt.plot(stel.sigma)
# plt.show()
# exit()
assert jnp.allclose(elongation, stel.elongation, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(inv_L_grad_B, stel.inv_L_grad_B, atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(sigma, stel.sigma, atol=a_tolerance, rtol=r_tolerance)
from qsc.calculate_r1 import _residual, _jacobian
stel.sigma0 = sigma[0]
assert jnp.allclose(res, _residual(stel, jnp.concatenate([jnp.array([iota]), sigma[1:]])), atol=a_tolerance, rtol=r_tolerance)
assert jnp.allclose(jac, _jacobian(stel, jnp.concatenate([jnp.array([iota]), sigma[1:]])), atol=a_tolerance, rtol=r_tolerance)