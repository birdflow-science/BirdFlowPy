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
    
# Every loss term shares this signature, even where it ignores most of
# the arguments, so terms are interchangeable in LOSS_TERMS below.
#     term_fn(pred_densities, flows, true_densities, d_matrices) -> scalar

def obs_loss(pred_densities, flows, true_densities, d_matrices):
    obs = 0
    for pred, true in zip(pred_densities, true_densities):
        residual = true - pred
        obs += jnp.sum(jnp.square(residual))
    return obs

def distance_loss(pred_densities, flows, true_densities, d_matrices):
    dist = 0
    for flow, d_matrix in zip(flows, d_matrices):
        dist += jnp.sum(flow * d_matrix)
    return dist

def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h

def ent_loss(pred_densities, flows, true_densities, d_matrices):
    ent = 0
    for f in flows:
        ent += entropy(f)
    for d in pred_densities:
        ent -= entropy(d)
    return ent

def flow_l2_loss(pred_densities, flows, true_densities, d_matrices):
    l2 = 0
    for flow in flows:
        l2 += jnp.sum(jnp.square(flow))
    return l2

# name -> (sign, term_fn). `sign` lets a term be rewarded (-1, like entropy,
# which we want to maximize) or penalized (+1) without hardcoding which is
# which in loss_fn. Adding a new loss term means adding one entry here.
LOSS_TERMS = {
    'obs': (1.0, obs_loss),
    'dist': (1.0, distance_loss),
    'ent': (-1.0, ent_loss),
    'flow_l2': (1.0, flow_l2_loss),
}

# obs/dist/ent are the three terms update_hdf.py's output filename has
# always encoded, regardless of whether they're at their default value.
HISTORIC_WEIGHT_NAMES = ('obs', 'dist', 'ent')

def extra_weight_filename_suffix(weights, defaults, historic_names=HISTORIC_WEIGHT_NAMES):
    """Build a filename suffix for loss weights beyond the historic
    obs/dist/ent trio, which update_hdf.py's output filename already
    encodes unconditionally.

    Any other weight (e.g. flow_l2) is included only if it differs from
    its registered default in `defaults` (each term's --<name>_weight
    CLI default), so a run that doesn't touch a newly added term
    produces the same filename it would have before that term existed --
    avoiding a silent filename collision between two runs that differ
    only in a weight that previously didn't exist.
    """
    parts = []
    for name in sorted(weights):
        if name in historic_names:
            continue
        if weights[name] != defaults.get(name, 0.0):
            parts.append(f'{name}{weights[name]}')
    return ''.join(f'_{p}' for p in parts)

def loss_fn(params, cells, true_densities, d_matrices, weights):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]

    total = 0
    components = {}
    for name, (sign, term_fn) in LOSS_TERMS.items():
        value = term_fn(pred_densities, flows, true_densities, d_matrices)
        components[name] = value
        total += sign * weights.get(name, 0.0) * value

    return total, components

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

    loss_dict = {'total': []}

    for step in range(training_steps):
        params, opt_state, loss = update(params, opt_state)
        total_loss, loss_components = loss
        loss_dict['total'].append(float(total_loss))
        for name, value in loss_components.items():
            loss_dict.setdefault(name, []).append(float(value))

    return params, loss_dict