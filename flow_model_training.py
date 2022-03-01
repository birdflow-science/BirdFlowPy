from flow_model import FlowModel, model_forward

import haiku as hk
import optax

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from scipy.spatial.distance import pdist, squareform

Datatuple = namedtuple('Datatuple', ['weeks', 'x_dim', 'y_dim', 'cells', 'nan_mask'])

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

def gen_d_matrix(xdim, ydim, nan_mask):
    x = jnp.linspace(0, xdim - 1, xdim)
    y = jnp.linspace(0, ydim - 1, ydim)
    xs, ys = jnp.meshgrid(x, y)
    xs = xs.flatten()
    ys = ys.flatten()
    coordinates = jnp.concatenate((xs, ys)).reshape((-1, 2), order='F')
    masked_coordinates = coordinates[~nan_mask]
    return squareform(pdist(masked_coordinates))
    
def obs_loss(pred_densities, true_densities):
    residual = true_densities - pred_densities
    return jnp.sum(jnp.square(residual))

def distance_loss(flows, d_matrix):
    return jnp.sum(flows * d_matrix)
  
def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h
  
vec_entropy = vmap(entropy)

def joint_entropy(joints, marginals):
    return jnp.sum(vec_entropy(joints)) - jnp.sum(vec_entropy(marginals[1:-1]))

def loss_fn(params, true_densities, d_matrix, obs_weight, dist_weight, ent_weight):
    weeks, cells = true_densities.shape[0], true_densities.shape[1]
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = jnp.array([d0] + [jnp.sum(flow, axis=0) for flow in flows])
    
    obs = obs_loss(pred_densities, true_densities)
    dist = distance_loss(flows, d_matrix)
    ent = joint_entropy(flows, pred_densities)
    
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