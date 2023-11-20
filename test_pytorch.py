from time import time
from qsc import Qsc
from main_pytorch import nacx_residual
import torch

device = torch.device("mps")

nfp = 2
rc = torch.tensor([1, 0.1, 0.01, 0.001], device=device)
zs = torch.tensor([0, 0.1, 0.01, 0.001], device=device)
sG = 1
spsi = 1
I2 = 0
B0 = 1
etabar = 0.9
sigma0=0.2
nphi = 31

rc_qsc = rc.cpu()
zs_qsc = zs.cpu()
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC values took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
start_time=time();stel = Qsc(rc_qsc, zs_qsc, etabar=etabar, nfp=nfp, nphi=nphi, sigma0=sigma0)
print('Calculating pyQSC again  took {} seconds'.format(time() - start_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, device=device, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp)
print('Calculating TORCH values took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, device=device, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp)
print('Calculating TORCH again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time();_ = nacx_residual(eR=rc, eZ=zs, etabar=etabar, device=device, nphi=nphi, sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp)
print('Calculating TORCH again  took {} seconds'.format(time() - intermediate_time))
intermediate_time = time()
torch_tangent_cylindrical, torch_normal_cylindrical, torch_binormal_cylindrical, \
    torch_curvature, torch_torsion, torch_G0, torch_axis_length, torch_varphi, \
        torch_d_d_varphi, res, sigma, iota, helicity, elongation, jac, inv_L_grad_B, torch_phi, torch_d_d_phi = nacx_residual(eR=rc, eZ=zs, etabar=etabar, device=device, nphi=nphi,
                                                                                      sigma0=sigma0, I2=I2, spsi=spsi, sG=sG, B0=B0, nfp=nfp)
print('Calculating TORCH again  took {} seconds'.format(time() - intermediate_time))
## Test that the TORCH values are the same as the pyQSC values
import numpy as np
a_tolerance = 1e-5
r_tolerance = 1e-5
np.testing.assert_allclose(torch_phi.cpu(), stel.phi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_tangent_cylindrical.cpu(), stel.tangent_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_G0.cpu(), stel.G0, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_curvature.cpu(), stel.curvature, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_normal_cylindrical.cpu(), stel.normal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_binormal_cylindrical.cpu(), stel.binormal_cylindrical, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_torsion.cpu(), stel.torsion, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_axis_length.cpu(), stel.axis_length, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_varphi.cpu(), stel.varphi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_d_d_phi.cpu(), stel.d_d_phi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(torch_d_d_varphi.cpu(), stel.d_d_varphi, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(helicity.cpu(), stel.helicity, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(elongation.cpu(), stel.elongation, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(inv_L_grad_B.cpu(), stel.inv_L_grad_B, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(sigma.cpu(), stel.sigma, atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(iota.cpu(), stel.iota, atol=a_tolerance, rtol=r_tolerance)
from qsc.calculate_r1 import _residual, _jacobian
stel.sigma0 = sigma[0].cpu()
np.testing.assert_allclose(res.cpu(), _residual(stel, np.concatenate([np.array([iota.cpu()]), sigma.cpu()[1:]])), atol=a_tolerance, rtol=r_tolerance)
np.testing.assert_allclose(jac.cpu(), _jacobian(stel, np.concatenate([np.array([iota.cpu()]), sigma.cpu()[1:]])), atol=a_tolerance, rtol=r_tolerance)