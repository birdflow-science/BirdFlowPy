from flow_model import FlowModel, model_forward
from flow_model_training import Datatuple, gen_d_matrix, loss_fn, train_model

import os
import pickle
from functools import partial
import argparse

import numpy as np
import h5py

import optax
import haiku as hk
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('root', type=str, help='hdf root directory')
parser.add_argument('species', type=str, help='species name')
parser.add_argument('resolution', type=int, help='model resolution')

args = parser.parse_args()

print(args.root, args.species, args.resolution)

rng_seed = 17
training_steps = 1500

learning_rate = 0.1
obs_weight = 20.0
ent_weight = 0.01
dist_weight = 0.2
dist_pow = 0.3

file = h5py.File(os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5'), 'r+')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
cells = true_densities.shape[1]

x_dim = int(np.asarray(file['geom']['ncol']))
y_dim = int(np.asarray(file['geom']['nrow']))

nan_mask = ~np.asarray(file['geom']['mask']).flatten().astype(bool)
dtuple = Datatuple(weeks, x_dim, y_dim, cells, nan_mask)

d_matrix = gen_d_matrix(dtuple.x_dim, dtuple.y_dim, dtuple.nan_mask)
d_matrix = d_matrix ** dist_pow

# Get the random seed and optimizer
key = hk.PRNGSequence(rng_seed)
optimizer = optax.adam(learning_rate)

# Instantiate loss function
loss_fn = jit(partial(loss_fn, 
                      true_densities=true_densities, 
                      d_matrix=d_matrix, 
                      obs_weight=obs_weight, 
                      dist_weight=dist_weight,
                      ent_weight=ent_weight))


# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                training_steps,
                                dtuple.cells,
                                dtuple.weeks,
                                key)

with open(os.path.join(args.root, f'{args.species}_params_{args.resolution}.pkl'), 'wb') as f:
     pickle.dump(params, f)
        
t_start = 1
t_end = len(params) # zero indexing in range and extra item cancel each other out

# Initial distribution
d = softmax(params["Flow_Model/Initial_Params"]["z0"])

# Calculate marginals  "flow_amounts"
flow_amounts = []
for week in range(t_start, t_end):
    z = params[f'Flow_Model/Week_{week}']['z']
    trans_prop = softmax(z, axis=1)  # softmax on rows
    flow = trans_prop * d.reshape(-1, 1) # convert d to a column and multiply each row in trans_prop by the corresponding scalar in d
    flow_amounts.append(flow)
    d = flow.sum(axis=0)
    
file.create_dataset('marginals', data = flow_amounts)
file.close() 