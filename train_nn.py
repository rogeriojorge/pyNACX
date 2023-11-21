#!/usr/bin/env python3
import jax
import optax
import json
import itertools
import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn
from flax import serialization
from main_jax import nacx_residual

learning_rate = 5e-4
n_training_steps = int(1e4)
nsamples = 21
number_of_retrainings = 6
eReZmax = 0.175
eReZmin = 1e-2
etabarMin = 0.3
etabarMax = 2.0
nfp = 2
nphi = 51
loss_tolerance = 1e-5
b1=0.9
b2=0.999
eps=1e-8
min_steps_to_take = 100

model_save_path = f'model_params_nfp{nfp}.bin'
loss_save_path = f'loss_history_nfp{nfp}.json'

number_of_x_parameters = 3
number_of_y_parameters = 3

rng = jax.random.PRNGKey(0)
rng1, rng2 = jax.random.split(rng)

### Define the model
@jit
def forward_solver(parameters):
    eR  = parameters[0]
    eZ  = parameters[1]
    eta = parameters[2]
    iota, e, iL = nacx_residual(jnp.array([1, -eR]), jnp.array([0, eZ]), eta, nfp=nfp, nphi=nphi)
    return jnp.array([iota, jnp.max(e), jnp.max(iL)])

@jit
def vmap_forward_solver(x_samples):
    result = vmap(forward_solver, in_axes=0)(x_samples)
    return result

class DeepNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return nn.Dense(number_of_y_parameters)(x)
model = DeepNN()

def make_mae_func(x_batched, y_batched):
    def mae(params):
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.abs(y - pred)
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched))
    return jax.jit(mae) 

tx = optax.chain(
    optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
    optax.scale(-learning_rate)
)

## Function to train the model
def train_model_with_data(x_samples, y_samples, params, loss_array=[]):
    loss = make_mae_func(x_samples, y_samples)
    loss_grad_fn = jax.value_and_grad(loss)
    opt_state = tx.init(params)
    loss_old=jnp.inf
    for step in range(n_training_steps):
        loss_val, grads = loss_grad_fn(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss_array.append(loss_val)
        if step % 300 == 0 or step == n_training_steps - 1:
            print(f'  Loss[{step}] = {loss_val}')
        if jnp.abs((loss_val - loss_old)/loss_old) < loss_tolerance and step>min_steps_to_take:
            print(f'  Loss[{step}] = {loss_val}')
            break
        loss_old = loss_val
    return params, opt_state, loss_array

if __name__ == "__main__":
    ## Train the model
    params = model.init(rng2, jnp.ones((number_of_x_parameters,)))
    loss_array=[]
    for i in range(number_of_retrainings):
        print(f'Training {i+1} of {number_of_retrainings} with {n_training_steps} max steps')
        x_samples = jnp.array(list(itertools.product(jax.random.uniform(rng1+1+i, minval=eReZmin, maxval=eReZmax, shape=(nsamples,)),
                                                    jax.random.uniform(rng1+2+i, minval=eReZmin, maxval=eReZmax, shape=(nsamples,)),
                                                    jax.random.uniform(rng1+3+i, minval=etabarMin, maxval=etabarMax, shape=(nsamples,)))))
        y_samples = vmap_forward_solver(x_samples)
        params, opt_state, loss_array = train_model_with_data(x_samples, y_samples, params, loss_array)

    ## Save results
    bytes_output = serialization.to_bytes(params)
    with open(model_save_path, 'wb') as f:
        f.write(bytes_output)
    with open(loss_save_path, 'w') as f:
        json.dump(jnp.array(loss_array).tolist(), f)
