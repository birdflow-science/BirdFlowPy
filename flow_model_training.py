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
#     term_fn(pred_densities, flows, true_densities, d_matrices, masks) -> scalar

def obs_loss(pred_densities, flows, true_densities, d_matrices, masks):
    obs = 0
    for pred, true in zip(pred_densities, true_densities):
        residual = true - pred
        obs += jnp.sum(jnp.square(residual))
    return obs

def distance_loss(pred_densities, flows, true_densities, d_matrices, masks):
    dist = 0
    for flow, d_matrix in zip(flows, d_matrices):
        dist += jnp.sum(flow * d_matrix)
    return dist

def entropy(probs):
    logp = jnp.log(probs)
    ent = probs * logp
    h = -1 * jnp.sum(ent)
    return h

def ent_loss(pred_densities, flows, true_densities, d_matrices, masks):
    ent = 0
    for f in flows:
        ent += entropy(f)
    for d in pred_densities:
        ent -= entropy(d)
    return ent

def flow_l2_loss(pred_densities, flows, true_densities, d_matrices, masks):
    l2 = 0
    for flow in flows:
        l2 += jnp.sum(jnp.square(flow))
    return l2

def smooth_loss(pred_densities, flows, true_densities, d_matrices, masks):
    """Second-difference (curvature) penalty on the model's own weekly
    marginals, used to discourage spurious week-to-week movement driven by
    noise in the input abundance rasters (issue #7).

    Equivalently, per cell: linearly extrapolate abundance from weeks t-1
    and t to predict week t+1, then square the difference between that
    extrapolated value and the actual (fitted) value at t+1. The loss is
    the sum of this squared difference across all cells and all interior
    weeks. A cell whose abundance changes at a constant rate (steady
    migratory drift) extrapolates perfectly and costs nothing; a
    noise-driven spike-then-revert does not extrapolate and costs more.

    Each week's marginal lives in that week's compact, dynamic-mask-sized
    index space; masks[t] (length total_cells, True at week t's active
    cells) embeds it into a common canonical cell index so differences
    across weeks compare the same physical cells. Cells outside a week's
    mask are treated as true zero density (not "unknown"), so every cell
    contributes a term in every week -- no separate valid-cell tracking is
    needed.

    Open boundary, not circular: no curvature term is computed at week 0
    or the last week. BirdFlowR::preprocess_species() already linearizes
    the annual cycle for full-year models by duplicating week 1's target
    distribution as an extra final week before this code ever sees the
    data, so the weeks this function receives are always a linear chain
    (annual wraparound is already encoded in that duplicated week, not
    something smooth_loss needs to reconstruct); the rarer non-annual
    subset-of-year case is a genuine linear span with no wrap either way.
    A circular wrap here would instead fabricate two spurious curvature
    terms with no real adjacency behind them.

    Deliberately left as an unnormalized sum, like its sibling terms
    (obs_loss, ent_loss, flow_l2_loss) -- see dev/known-issues.md.
    """
    total_cells = masks.shape[1]
    weeks = len(pred_densities)

    embedded = []
    for t in range(weeks):
        full = jnp.zeros(total_cells).at[masks[t]].set(pred_densities[t])
        embedded.append(full)

    smooth = 0
    for t in range(1, weeks - 1):
        curvature = embedded[t - 1] - 2 * embedded[t] + embedded[t + 1]
        smooth += jnp.sum(jnp.square(curvature))
    return smooth

# name -> (sign, term_fn). `sign` lets a term be rewarded (-1, like entropy,
# which we want to maximize) or penalized (+1) without hardcoding which is
# which in loss_fn. Adding a new loss term means adding one entry here.
LOSS_TERMS = {
    'obs': (1.0, obs_loss),
    'dist': (1.0, distance_loss),
    'ent': (-1.0, ent_loss),
    'flow_l2': (1.0, flow_l2_loss),
    'smooth': (1.0, smooth_loss),
}

# obs/dist/ent are the three terms update_hdf.py's output filename has
# always encoded, regardless of whether they're zero or nonzero.
HISTORIC_WEIGHT_NAMES = ('obs', 'dist', 'ent')

def extra_weight_filename_suffix(weights, historic_names=HISTORIC_WEIGHT_NAMES):
    """Build a filename suffix for loss weights beyond the historic
    obs/dist/ent trio, which update_hdf.py's output filename already
    encodes unconditionally.

    Any other weight (e.g. flow_l2) is included only if it's nonzero --
    matching loss_fn's own weights.get(name, 0.0) convention, where a
    zero weight means the term has no effect on the loss at all, so a
    run that leaves a newly added term at zero produces the same
    filename it would have before that term existed. This avoids a
    silent filename collision between two runs that differ only in a
    weight that previously didn't exist.
    """
    parts = []
    for name in sorted(weights):
        if name in historic_names:
            continue
        if weights[name] != 0.0:
            parts.append(f'{name}{weights[name]}')
    return ''.join(f'_{p}' for p in parts)

def loss_fn(params, cells, true_densities, d_matrices, weights, masks):
    weeks = len(true_densities)
    pred = model_forward.apply(params, None, cells, weeks)
    d0, flows = pred
    pred_densities = [d0] + [jnp.sum(flow, axis=0) for flow in flows]

    total = 0
    components = {}
    for name, (sign, term_fn) in LOSS_TERMS.items():
        value = term_fn(pred_densities, flows, true_densities, d_matrices, masks)
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