import tensorflow as tf
from main_tensorflow import nacx_residual
from time import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
from qsc import Qsc

nfp = 2
nfourier = 6
etabar = 0.9
iota_min = 0.41
nphi = 71


rc_in = tf.pow(10.0, -tf.range(nfourier, dtype=tf.float32))
zs_in = tf.concat([tf.constant([0.0], dtype=tf.float32), tf.pow(10.0, -tf.range(1, nfourier, dtype=tf.float32))], axis=0)

def objective_function(params):
    length = tf.shape(params)[0]
    etabar = params[-1]
    length_of_each = (length - 1) // 2
    rc = tf.concat([tf.constant([1.0], dtype=tf.float32), params[:length_of_each]], axis=0)
    zs = tf.concat([tf.constant([0.0], dtype=tf.float32), params[length_of_each:-1]], axis=0)
    
    # Assuming that nacx_residual is a TensorFlow function
    iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)
    
    residuals = tf.concat([inv_L_grad_B, elongation, [1e2 * tf.abs(tf.maximum(0.0, 1.0 - tf.abs(iota) / iota_min))]], axis=0)
    return residuals
    # return tf.sum(residuals**2)

def grad_objective_function(params):
    jac = tf.GradientTape.jacobian(objective_function, params)
    return jac

print("Optimizing with {} Fourier modes".format(nfourier))
initial_params = tf.concat([rc_in[1:], zs_in[1:], [etabar]], axis=0)
print("Computing objective function and gradient")
start_time = time()
print(f'Initial sum of objective function: {np.sum(objective_function(initial_params)**2):.1e} took {(time() - start_time):.1e} seconds')
start_time = time()
print(f'Initial sum of grad of objective function: {np.sum(grad_objective_function(initial_params)**2):.1e} took {(time() - start_time):.1e} seconds')


def objective_function_jac(params):
    jac = grad_objective_function(params).numpy()
    return jac

def objective_function_np(params):
    return objective_function(params).numpy()
start_time = time()
result = least_squares(objective_function_np, initial_params.numpy(), jac=objective_function_jac, verbose=2, method='lm', x_scale='jac', max_nfev=int(5e3))


# def objective_function_np(params):
#     params = torch.tensor(params, device=device, requires_grad=True)
#     return objective_function(params).detach().numpy()
# def objective_function_jac(params):
#     params = torch.tensor(params, device=device, requires_grad=True)
#     return grad_objective_function(params).detach().numpy()
# start_time = time()
# result = minimize(objective_function_np, initial_params.detach().numpy(), jac=objective_function_jac, method='BFGS', options={'disp': True, 'maxiter': 1000})

print('Optimization took {} seconds'.format(time() - start_time))
optimized_params = result.x

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

rc = tf.concat([tf.constant([1.0], dtype=tf.float32), optimized_params[:len(rc_in) - 1]], axis=0)
zs = tf.concat([tf.constant([0.0], dtype=tf.float32), optimized_params[len(rc_in) - 1:len(rc_in) + len(zs_in) - 2]], axis=0)
etabar = optimized_params[-1]
iota, elongation, inv_L_grad_B = nacx_residual(eR=rc, eZ=zs, etabar=etabar, nphi=nphi, nfp=nfp)

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
