import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
import flax
import random
import collections
from main_jax import nacx_residual
from jax import jit, vmap
import itertools
import optax

# Constants from your original script
eReZmax = 0.17
eReZmin = 1e-2
etabarMin = 0.05
etabarMax = 3.0
nfp = 2
nphi = 51
nsamples = 23

rng = jax.random.PRNGKey(0)
rng1, rng2 = jax.random.split(rng)

# Define the neural network model
class InverseSolverModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # Assuming 3 parameters to predict
        return x

# Define the RL environment
class InverseSolverEnv:
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.data[self.current_index][1]  # Return the first y

    def step(self, action):
        true_x = self.data[self.current_index][0]
        reward = -jnp.linalg.norm(true_x - action)  # Negative L2 distance as reward
        self.current_index += 1
        done = self.current_index == len(self.data)
        next_state = self.data[self.current_index][1] if not done else None
        return next_state, reward, done

# Forward solver definition
@jit
def forward_solver(parameters):
    eR, eZ, eta = parameters
    iota, e, iL = nacx_residual(jnp.array([1, -eR]), jnp.array([0, eZ]), eta, nfp=nfp, nphi=nphi)
    return jnp.array([iota, jnp.max(e), jnp.max(iL)])

@jit
def vmap_forward_solver(x_samples):
    return vmap(forward_solver, in_axes=0)(x_samples)

# Data generation function
def generate_data(n_samples):
    y_samples = jnp.array(list(itertools.product(
        jax.random.uniform(rng1, minval=eReZmin, maxval=eReZmax, shape=(n_samples,)),
        jax.random.uniform(rng2, minval=eReZmin, maxval=eReZmax, shape=(n_samples,)),
        jax.random.uniform(rng1, minval=etabarMin, maxval=etabarMax, shape=(n_samples,))
    )))
    x_samples = vmap_forward_solver(y_samples)
    return list(zip(x_samples, y_samples))

# DQN Agent
class DQNAgent:
    def __init__(self, model, learning_rate=1e-3, gamma=0.99, batch_size=32, buffer_size=10000, epsilon=0.1):
        self.model = model
        self.optimizer = optax.adam(learning_rate).create(model.init(rng2, jnp.ones((3,))))
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = collections.deque(maxlen=buffer_size)
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return jax.random.normal(jax.random.PRNGKey(0), (3,))
        else:
            return self.model.apply(self.optimizer.target, state)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Compute loss and gradients
        def loss_fn(params):
            q_values = self.model.apply(params, jnp.array(states))
            next_q_values = self.model.apply(params, jnp.array(next_states))
            max_next_q_values = jnp.max(next_q_values, axis=1)
            target_q_values = jnp.array(rewards) + self.gamma * max_next_q_values * (1 - jnp.array(dones))
            td_error = jnp.square(target_q_values - jnp.sum(q_values * jnp.array(actions), axis=1))
            return jnp.mean(td_error)

        grads = jax.grad(loss_fn)(self.optimizer.target)
        self.optimizer = self.optimizer.apply_gradient(grads)

    def replay_buffer_add(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

# Training loop
def train(model, environment, agent, num_episodes):
    for episode in range(num_episodes):
        state = environment.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = environment.step(action)
            agent.replay_buffer_add(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Main execution
if __name__ == "__main__":
    n_samples = 3 #1000
    data = generate_data(n_samples)
    env = InverseSolverEnv(data)
    model = InverseSolverModel()
    agent = DQNAgent(model)
    train(model, env, agent, 100)

# Save and load model functions
def save_model(model, filename):
    checkpoints.save_checkpoint("./", model, step=0, prefix=filename)

def load_model(filename):
    return checkpoints.restore_checkpoint("./", target=None, prefix=filename)
