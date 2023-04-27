from flow_model import FlowModel, model_forward

import haiku as hk
import optax

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from scipy.spatial.distance import pdist, squareform

Datatuple = namedtuple('Datatuple', ['weeks', 'cells', 'distances', 'masks'])

def process_data(data_array):
    weeks, y_dim, x_dim = data_array.shape
    
    flat_data_array = data_array.reshape(weeks, -1)
    nans = jnp.isnan(flat_data_array[0])

    mass = flat_data_array[:, ~nans]
    reg = mass.sum(axis=1)
    density = mass / reg[:, None]

    cells = density.shape[1]
    dtuple = Datatuple(weeks, x_dim, y_dim, cells, nans)
    return density, dtuple


def mask_input(true_densities, dtuple):
    distance_matrix = jnp.zeros((dtuple.cells, dtuple.cells))
    distance_matrix = distance_matrix.at[jnp.triu_indices(dtuple.cells, k=1)].set(dtuple.distances)
    distance_matrix = distance_matrix + distance_matrix.T

    distance_matrices = []
    for i in range(0, dtuple.weeks - 1):
        distance_matrices.append(distance_matrix[dtuple.masks[i], :][:, dtuple.masks[i + 1]])

    masked_densities = []
    for density, mask in zip(true_densities, dtuple.masks):
        masked_densities.append(density[mask])
        
    return distance_matrices, masked_densities
    
def obs_loss(pred_densities, true_densities):
    obs = 0
    for pred, true in zip(pred_densities, true_densities):
        residual = true - pred
        obs += jnp.sum(jnp.square(residual))
    return obs

def distance_loss(flows, d_matrices):
    dist = 0
    for flow, d_matrix in zip(flows, d_matrices):
        dist += jnp.sum(flow * d_matrix)
    return dist

def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h

def ent_loss(probs, flows):
    ent = 0
    for p in probs:
        ent += entropy(p)
    for f in flows:
        ent -= entropy(f)
    return ent

def loss_fn(params, cells, true_densities, d_matrices, obs_weight, dist_weight, ent_weight):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]
    
    obs = obs_loss(pred_densities, true_densities)
    dist = distance_loss(flows, d_matrices)
    ent = ent_loss(flows, pred_densities)
    
    return (obs_weight * obs) + (dist_weight * dist) + (-1 * ent_weight * ent), (obs, dist, ent)

def train_model(loss_fn,
                optimizer,
                training_steps,
                cells,
                weeks,
                key):
    params = model_forward.init(next(key), cells, weeks)
    opt_state = optimizer.init(params)

    def update(params, opt_state):
        loss, grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    update = jit(update)

    loss_dict = {
        'total' : [],
        'obs' : [],
        'dist' : [],
        'ent' : [],
    }

    for step in range(training_steps):
        params, opt_state, loss = update(params, opt_state)
        total_loss, loss_components = loss
        obs, dist, ent = loss_components
        loss_dict['total'].append(float(total_loss))
        loss_dict['obs'].append(float(obs))
        loss_dict['dist'].append(float(dist))
        loss_dict['ent'].append(float(ent))
    
    return params, loss_dict