import tensorflow as tf
import numpy as np
from qsc import Qsc
from time import time
from main_tensorflow import nacx_residual

nfp = 2
rc = tf.constant([1, 0.1, 0.01, 0.001], dtype=tf.float32)
zs = tf.constant([0, 0.1, 0.01, 0.001], dtype=tf.float32)
etabar = tf.constant(0.9, dtype=tf.float32)
nphi = 31

rc_qsc = rc.numpy()
zs_qsc = zs.numpy()
etabar_qsc = etabar.numpy()

start_time = time()
stel = Qsc(rc_qsc, zs_qsc, etabar=etabar_qsc, nfp=nfp, nphi=nphi)
print('Calculating pyQSC values took {} seconds'.format(time() - start_time))

start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar_qsc, nfp=nfp, nphi=nphi)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar_qsc, nfp=nfp, nphi=nphi)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar_qsc, nfp=nfp, nphi=nphi)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
print('Calculating tf values took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
print('Calculating tf again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp, debug=True)
print('Calculating tf again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
print('Calculating tf again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time()
tf_tangent_cylindrical, tf_normal_cylindrical, tf_binormal_cylindrical, \
    tf_curvature, tf_torsion, tf_G0, tf_axis_length, tf_varphi, \
        tf_d_d_varphi, res, sigma, iota, helicity, elongation, jac, inv_L_grad_B, tf_phi, tf_d_d_phi = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp, debug=True)
print('Calculating tf again  took {} seconds'.format(time() - intermediate_time))
## Test that the tf values are the same as the pyQSC values
import numpy as np
a_tolerance = 1e-5
r_tolerance = 1e-5
np.testing.assert_allclose(tf_phi.numpy(), stel.phi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_tangent_cylindrical.numpy(), stel.tangent_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_G0.numpy(), stel.G0, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_curvature.numpy(), stel.curvature, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_normal_cylindrical.numpy(), stel.normal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_binormal_cylindrical.numpy(), stel.binormal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_torsion.numpy(), stel.torsion, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_axis_length.numpy(), stel.axis_length, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_varphi.numpy(), stel.varphi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_d_d_phi.numpy(), stel.d_d_phi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(tf_d_d_varphi.numpy(), stel.d_d_varphi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(helicity.numpy(), stel.helicity, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(elongation.numpy(), stel.elongation, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(inv_L_grad_B.numpy(), stel.inv_L_grad_B, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(sigma.numpy(), stel.sigma, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(iota.numpy(), stel.iota, atol=a_tolerance, rtol=r_tolerance)
from qsc.calculate_r1 import _residual, _jacobian
stel.sigma0 = sigma[0].numpy()
np.testing.assert_allclose(res.numpy(), _residual(stel, np.concatenate([np.array([iota.numpy()]), sigma.numpy()[1:]])), atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(jac.numpy(), _jacobian(stel, np.concatenate([np.array([iota.numpy()]), sigma.numpy()[1:]])), atol=a_tolerance, rtol=r_tolerance)