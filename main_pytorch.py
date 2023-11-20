import torch
from torch import jit
import math

#@jit.script
def compute_terms(jn: int, phi: torch.Tensor, n_values: torch.Tensor, eR: torch.Tensor, eZ: torch.Tensor):
    n = n_values[jn]
    sinangle = torch.sin(n * phi)
    cosangle = torch.cos(n * phi)
    return torch.stack([eR[jn] * cosangle, eZ[jn] * sinangle,
                        eR[jn] * (-n * sinangle), eZ[jn] * (n * cosangle),
                        eR[jn] * (-n * n * cosangle), eZ[jn] * (-n * n * sinangle),
                        eR[jn] * (n * n * n * sinangle), eZ[jn] * (-n * n * n * cosangle)])

#@jit.script
def spectral_diff_matrix(nphi: int, nfp: int, device: torch.device):
    pi = torch.tensor(math.pi)
    xmin = 0
    xmax = 2 * pi / nfp
    h = 2 * pi / nphi
    kk = torch.arange(1, nphi, device=device)
    n_half = nphi // 2
    topc = 1 / torch.sin(torch.arange(1, n_half + 1, device=device) * h / 2)
    temp = torch.cat((topc, topc.flip(dims=[0])))
    col1 = torch.cat((torch.tensor([0], device=device), 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    vals = torch.cat((row1.flip([0]), col1[1:]))
    a = torch.arange(len(col1)).unsqueeze(1)
    b = torch.arange(len(row1) - 1, -1, -1).unsqueeze(1).T
    return 2 * pi / (xmax - xmin) * vals[a + b]

#@jit.script
def determine_helicity(normal_cylindrical: torch.Tensor, spsi: float, sG: float):
    x_positive = normal_cylindrical[:, 0] >= 0
    z_positive = normal_cylindrical[:, 2] >= 0
    quadrant = 1 * x_positive * z_positive + 2 * ~x_positive * z_positive \
             + 3 * ~x_positive * ~z_positive + 4 * x_positive * ~z_positive
    quadrant = torch.cat((quadrant, quadrant[0:1]))
    delta_quadrant = quadrant[1:] - quadrant[:-1]
    increment = torch.sum((quadrant[:-1] == 4) & (quadrant[1:] == 1))
    decrement = torch.sum((quadrant[:-1] == 1) & (quadrant[1:] == 4))
    return (torch.sum(delta_quadrant) + increment - decrement) * spsi * sG

#@jit.script
def replace_first_element(x: torch.Tensor, new_value: float, device: torch.device):
    return torch.cat([torch.tensor([new_value], device=device), x[1:]])

#@jit.script
def sigma_equation_residual(x: torch.Tensor, sigma0: float, curvature: torch.Tensor, etabar: torch.Tensor, 
                            spsi: float, torsion: torch.Tensor, I2: float, B0: float, G0: float, 
                            helicity: float, nfp: int, d_d_varphi: torch.Tensor, device: torch.device):
    iota = x[0]
    sigma = replace_first_element(x, sigma0, device)
    etaOcurv2 = etabar**2 / curvature**2
    return torch.matmul(d_d_varphi, sigma) \
    + (iota + helicity * nfp) * (etaOcurv2**2 + 1 + sigma**2) \
    - 2 * etaOcurv2 * (-spsi * torsion + I2 / B0) * G0 / B0

#@jit.script
def sigma_equation_jacobian(x: torch.Tensor, sigma0: float, curvature: torch.Tensor, etabar: torch.Tensor, 
                            helicity: float, nfp: int, d_d_varphi: torch.Tensor, device: torch.device):
    iota = x[0]
    sigma = replace_first_element(x, sigma0, device)
    etaOcurv2 = etabar**2 / curvature**2
    fill_value = (etaOcurv2**2 + 1 + sigma**2).unsqueeze(1)
    jac = d_d_varphi + (iota + helicity * nfp) * 2 * torch.diag(sigma)
    # return jac.index_fill_(1, torch.tensor(0, device=device), fill_value)
    jac[:, 0] = fill_value.squeeze()
    return jac

#@jit.script
def newton(sigma0: float, curvature: torch.Tensor, etabar: torch.Tensor, 
           spsi: float, torsion: torch.Tensor, I2: float, B0: float, G0: float, 
           helicity: float, nfp: int, d_d_varphi: torch.Tensor, device: torch.device):
    x0 = torch.full((len(torsion),), sigma0, device=device)
    x0 = replace_first_element(x0, 0., device)

    niter = 5
    for _ in range(niter):
        residual = sigma_equation_residual(x0, sigma0, curvature, etabar, spsi, torsion, I2, B0, G0, helicity, nfp, d_d_varphi, device)
        jacobian = sigma_equation_jacobian(x0, sigma0, curvature, etabar, helicity, nfp, d_d_varphi, device)
        # step = torch.linalg.solve(jacobian, -residual.unsqueeze(1)).squeeze(1)
        step = -torch.inverse(jacobian).matmul(residual.unsqueeze(1)).squeeze(1)
        ### x0 = x0 + step
        x0.add_(step)

    return x0

# @jit.script
def nacx_residual(eR: torch.Tensor, eZ: torch.Tensor, etabar: torch.Tensor,
                  device: torch.device, nphi: int, nfp: int):
    spsi=1.;I2=0.;sG=1.;B0=1.;sigma0=0.
    assert nphi % 2 == 1, 'nphi must be odd'
    pi = torch.tensor(math.pi)
    phi = torch.linspace(0, 2 * pi / nfp, nphi+1, device=device)[:-1]
    d_phi = phi[1] - phi[0]
    nfourier = max(len(eR), len(eZ))
    n_values = torch.arange(nfourier, device=device) * nfp
    # Next are the arrays R0, Z0, R0p, Z0p, R0pp, Z0pp, R0ppp, Z0ppp
    R0Z0ppp = torch.sum(torch.stack([compute_terms(i, phi, n_values, eR, eZ) for i in range(nfourier)]), dim=0)
    d_l_d_phi = torch.sqrt(R0Z0ppp[0] * R0Z0ppp[0] + R0Z0ppp[2] * R0Z0ppp[2] + R0Z0ppp[3] * R0Z0ppp[3])
    d2_l_d_phi2 = (R0Z0ppp[0] * R0Z0ppp[2] + R0Z0ppp[2] * R0Z0ppp[4] + R0Z0ppp[3] * R0Z0ppp[5]) / d_l_d_phi
    B0_over_abs_G0 = nphi / torch.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    d_l_d_varphi = abs_G0_over_B0
    G0 = sG * abs_G0_over_B0 * B0
    d_r_d_phi_cylindrical = torch.stack([R0Z0ppp[2], R0Z0ppp[0], R0Z0ppp[3]], dim=1)
    d2_r_d_phi2_cylindrical = torch.stack([R0Z0ppp[4] - R0Z0ppp[0], 2 * R0Z0ppp[2], R0Z0ppp[5]], dim=1)
    d3_r_d_phi3_cylindrical = torch.stack([R0Z0ppp[6] - 3 * R0Z0ppp[2], 3 * R0Z0ppp[4] - R0Z0ppp[0], R0Z0ppp[7]], dim=1)
    tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi.unsqueeze(1)
    d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2.unsqueeze(1) / d_l_d_phi.unsqueeze(1) \
                                 + d2_r_d_phi2_cylindrical) / (d_l_d_phi.unsqueeze(1)**2)
    curvature = torch.sqrt(torch.sum(d_tangent_d_l_cylindrical**2, dim=1))
    axis_length = torch.sum(d_l_d_phi) * d_phi * nfp
    varphi = torch.cat([torch.zeros(1, device=device), torch.cumsum(d_l_d_phi[:-1] + d_l_d_phi[1:], dim=0)]) * (0.5 * d_phi * 2 * pi / axis_length)
    normal_cylindrical = d_tangent_d_l_cylindrical / curvature.unsqueeze(1)
    binormal_cylindrical = torch.linalg.cross(tangent_cylindrical, normal_cylindrical)
    torsion_numerator = torch.sum(d_r_d_phi_cylindrical * torch.linalg.cross(d2_r_d_phi2_cylindrical, 
                                                                      d3_r_d_phi3_cylindrical), dim=1)
    torsion_denominator = torch.sum(torch.linalg.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical)**2, dim=1)
    torsion = torsion_numerator / torsion_denominator
    d_d_phi = spectral_diff_matrix(nphi, nfp, device)
    d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    d_d_varphi = d_d_phi / d_varphi_d_phi.unsqueeze(1)
    helicity = determine_helicity(normal_cylindrical, spsi, sG)
    sigma = newton(sigma0, curvature, etabar, spsi, torsion, I2, B0, G0, helicity, nfp, d_d_varphi, device)
    iota = sigma[0]
    iotaN = iota + helicity * nfp
    sigma = replace_first_element(sigma, sigma0, device)
    x = torch.cat([iota.unsqueeze(0), sigma[1:].float()])
    res = sigma_equation_residual(x, sigma0, curvature, etabar, spsi, torsion, I2, B0, G0, helicity, nfp, d_d_varphi, device)
    jac = sigma_equation_jacobian(x, sigma0, curvature, etabar, helicity, nfp, d_d_varphi, device)
    X1c = etabar / curvature
    Y1s = sG * spsi * curvature / etabar
    Y1c = sG * spsi * curvature * sigma / etabar
    p =  X1c * X1c + Y1s * Y1s + Y1c * Y1c
    q = -X1c * Y1s
    elongation = (p + torch.sqrt(p * p - 4 * q * q)) / (2 * torch.abs(q))
    d_X1c_d_varphi = torch.matmul(d_d_varphi, X1c)
    d_Y1s_d_varphi = torch.matmul(d_d_varphi, Y1s)
    d_Y1c_d_varphi = torch.matmul(d_d_varphi, Y1c)
    factor = spsi * B0 / d_l_d_varphi
    tn = sG * B0 * curvature
    nt = tn
    bb = factor * (X1c * d_Y1s_d_varphi - iotaN * X1c * Y1c)
    nn = factor * (d_X1c_d_varphi * Y1s + iotaN * X1c * Y1c)
    bn = factor * (-sG * spsi * d_l_d_varphi * torsion - iotaN * X1c * X1c)
    nb = factor * (d_Y1c_d_varphi * Y1s - d_Y1s_d_varphi * Y1c + sG * spsi * d_l_d_varphi * torsion + iotaN * (Y1s * Y1s + Y1c * Y1c))
    grad_B_colon_grad_B = tn * tn + nt * nt + bb * bb + nn * nn + nb * nb + bn * bn
    L_grad_B = B0 * torch.sqrt(2 / grad_B_colon_grad_B)
    inv_L_grad_B = 1.0 / L_grad_B
    # return tangent_cylindrical, normal_cylindrical, binormal_cylindrical, curvature, torsion, G0, axis_length, varphi, d_d_varphi, res, sigma, iota, helicity, elongation, jac, inv_L_grad_B, phi, d_d_phi
    return iota, elongation, inv_L_grad_B
