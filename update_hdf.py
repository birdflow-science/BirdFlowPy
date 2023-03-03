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
parser.add_argument('--obs_weight', help='Weight on the observation term of the loss', default=20.0, type=float)
parser.add_argument('--dist_weight', help='Weight on the distance penalty in the loss', default=0.1, type=float)
parser.add_argument('--ent_weight', help='Weight on the joint entropy of the model', default=0.02, type=float)
parser.add_argument('--dist_pow', help='The exponent of the distance penalty', default=0.5, type=float)
parser.add_argument('--learning_rate', help='Learning rate for Adam optimizer', default=0.1, type=float)
parser.add_argument('--training_steps', help='The number of training iterations', default=1500, type=int)
parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)

args = parser.parse_args()

print(str(args))

file = h5py.File(os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5'), 'r+')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
cells = true_densities.shape[1]

x_dim = int(np.asarray(file['geom']['ncol']))
y_dim = int(np.asarray(file['geom']['nrow']))

nan_mask = ~np.asarray(file['geom']['mask']).flatten().astype(bool)
dtuple = Datatuple(weeks, x_dim, y_dim, cells, nan_mask)

d_matrix = gen_d_matrix(dtuple.x_dim, dtuple.y_dim, dtuple.nan_mask)
d_matrix = d_matrix ** args.dist_pow

# Get the random seed and optimizer
key = hk.PRNGSequence(args.rng_seed)
optimizer = optax.adam(args.learning_rate)

# Instantiate loss function
loss_fn = jit(partial(loss_fn, 
                      true_densities=true_densities, 
                      d_matrix=d_matrix, 
                      obs_weight=args.obs_weight, 
                      dist_weight=args.dist_weight,
                      ent_weight=args.ent_weight))


# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                args.training_steps,
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