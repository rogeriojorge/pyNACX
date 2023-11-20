import torch
from main_pytorch import nacx_residual
from time import time
import numpy as np
from scipy.optimize import minimize
from qsc import Qsc
# device = torch.device("mps")
device = torch.device("cpu")

nfp = 2
nfourier = 6
etabar = torch.tensor([0.9], device=device)
iota_min = 0.41
nphi = 71

rc_in = torch.pow(10.0, -torch.arange(nfourier).float()).to(device)
zs_in = torch.cat([torch.tensor([0.0]), torch.pow(10.0, -torch.arange(1, nfourier).float())]).to(device)

def objective_function(params):
    length = params.size(0)
    etabar = params[-1]
    length_of_each = (length - 1) // 2
    rc = torch.cat([torch.tensor([1.0], device=device), params[:length_of_each]])
    zs = torch.cat([torch.tensor([0.0], device=device), params[length_of_each:-1]])
    iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, device=device, nphi=nphi, nfp=nfp)
    residuals = torch.cat([inv_L_grad_B, elongation, 1e2 * torch.abs(torch.max(torch.tensor([0.0], device=device), 1 - torch.abs(iota)/iota_min))])
    return residuals
    # return torch.sum(residuals**2)

def grad_objective_function(params):
    jac = torch.func.jacfwd(objective_function)(params)
    return jac

print("Optimizing with {} fourier modes".format(nfourier))
initial_params = torch.cat([rc_in[1:], zs_in[1:], etabar]).requires_grad_(True)
print("Computing objective function and gradient")
start_time = time()
print(f'Initial sum of objective function: {torch.sum(objective_function(initial_params)**2):.1e} took {(time() - start_time):.1e} seconds')
start_time = time()
print(f'Initial sum of grad of objective function: {torch.sum(grad_objective_function(initial_params)**2):.1e} took {(time() - start_time):.1e} seconds')


import numpy as np
from scipy.optimize import least_squares
def objective_function_jac(params):
    params = torch.tensor(params, device=device, requires_grad=True)
    return grad_objective_function(params).detach().numpy()
def objective_function_np(params):
    params = torch.tensor(params, device=device, requires_grad=True)
    return objective_function(params).detach().numpy()
start_time = time()
result = least_squares(objective_function_np, initial_params.detach().numpy(), jac=objective_function_jac, verbose=2, method='lm', x_scale='jac', max_nfev=int(5e3))

# def objective_function_np(params):
#     params = torch.tensor(params, device=device, requires_grad=True)
#     return objective_function(params).detach().numpy()
# def objective_function_jac(params):
#     params = torch.tensor(params, device=device, requires_grad=True)
#     return grad_objective_function(params).detach().numpy()
# start_time = time()
# result = minimize(objective_function_np, initial_params.detach().numpy(), jac=objective_function_jac, method='BFGS', options={'disp': True, 'maxiter': 1000})

print('Optimization took {} seconds'.format(time() - start_time))
optimized_params = torch.tensor(result.x)

# def closure():
#     optimizer.zero_grad()
#     loss = objective_function(initial_params)
#     loss.backward()
#     return loss
# # optimizer = torch.optim.Adam([initial_params], lr=1e-4)
# num_iterations = 1000
# optimizer = torch.optim.Adamax([initial_params], lr=6e-3)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=num_iterations)
# old_loss = float('inf')
# for iteration in range(num_iterations):
#     loss = optimizer.step(closure)
#     scheduler.step()
#     if iteration % 1 == 0:
#         print(f'Iteration {iteration}, Loss: {loss.item()}, Old loss: {old_loss}')
#     if (old_loss - loss.item()) / (loss.item()+1e-7) < 1e-8 and iteration > 100:
#         break
#     old_loss = loss.item()

# # Extracting the optimized parameters
# optimized_params = initial_params.detach()

rc = torch.cat([torch.tensor([1.0]), optimized_params[0:len(rc_in)-1]])
zs = torch.cat([torch.tensor([0.0]), optimized_params[len(rc_in)-1:len(rc_in)+len(zs_in)-2]])
etabar = optimized_params[-1]
iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp, device=device)

print('Optimized rc: {}'.format(rc.cpu().numpy()))
print('Optimized zs: {}'.format(zs.cpu().numpy()))
print('Optimized etabar: {}'.format(etabar.cpu().numpy()))
print('Optimized iota: {}'.format(iota.cpu().numpy()))
print('Optimized min inv_L_grad_B: {}'.format(np.min(inv_L_grad_B.cpu().numpy())))
print('Optimized max elongation: {}'.format(np.max(elongation.cpu().numpy())))
stel = Qsc(rc=rc, zs=zs, nfp=nfp, nphi=nphi, etabar=etabar.cpu().numpy())
print(f'True iota: {stel.iota}')
print(f'True min inv_L_grad_B: {np.min(stel.inv_L_grad_B)}')
print(f'True max elongation: {np.max(stel.elongation)}')
# stel.plot()
# stel.plot_boundary(r=0.1)
# exit()
