import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap
import optax
import flax.linen as nn
from main_jax import nacx_residual
from functools import partial
import matplotlib.pyplot as plt

# Input parameters
nfp = 3
nphi = 31
min_eReZ = -0.3
max_eReZ = 0.3
min_eta = -3
max_eta = 3
num_epochs = 5000
batch_size = 31
learning_rate = 1e-3
random_key = jax.random.PRNGKey(6438)

# Neural network architecture
class InverseSolverNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        return nn.Dense(3)(x)

# Forward solver
def forward_solver(params):
    eR, eZ, eta = params
    iota, e, iL = nacx_residual(jnp.array([1, eR]), jnp.array([0, eZ]), eta, nfp=nfp, nphi=nphi)
    return jnp.array([iota, jnp.max(e), jnp.max(iL)])

# Loss function
@partial(jit, static_argnums=(3,))
def loss_fn(params, x, y, model):
    pred = model.apply(params, x)
    return jnp.mean(jnp.square(pred - jnp.asarray(y)))

# JIT-compiled part of the training step
@partial(jit, static_argnums=(3,))
def compute_loss_and_grads(params, x, y, model):
    return value_and_grad(loss_fn)(params, x, y, model)

# Training step
def train_step(opt_state, params, x, y, model):
    loss, grads = compute_loss_and_grads(params, x, y, model)
    updates, new_opt_state = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# Generate training data
def generate_training_data(random_key, batch_size):
    random_key, subkey = jax.random.split(random_key)
    eR = jax.random.uniform(subkey, minval=min_eReZ, maxval=max_eReZ, shape=(batch_size,))
    random_key, subkey = jax.random.split(random_key)
    eZ = jax.random.uniform(subkey, minval=min_eReZ, maxval=max_eReZ, shape=(batch_size,))
    random_key, subkey = jax.random.split(random_key)
    eta = jax.random.uniform(subkey, minval=min_eta, maxval=max_eta, shape=(batch_size,))
    params = jnp.stack([eR, eZ, eta], axis=1)
    outputs = vmap(forward_solver, in_axes=0)(params)
    return random_key, params, outputs

# Initialize model and optimizer
model = InverseSolverNN()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)

# Training loop with model passed to train_step
losses = []
for epoch in range(num_epochs):
    random_key, x, y = generate_training_data(random_key, batch_size)
    params, opt_state, loss = train_step(opt_state, params, x, y, model)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Plotting the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Inverse solver function
def inverse_solver(iota, e, iL):
    x = jnp.array([iota, e, iL])
    prediction = model.apply(params, x)
    return prediction

# Example usage
iota, e, iL = 0.1, 10.0, 10.0
predicted_params = inverse_solver(iota, e, iL)
print(f"Predicted Parameters: {predicted_params}")
print(f"Asked iota, e, L: {iota, e, iL}")
print(f"True iota, e, L: {forward_solver(predicted_params)}")
