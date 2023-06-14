from flow_model import FlowModel, model_forward
from flow_model_training import Datatuple, mask_input, loss_fn, train_model

import os
import shutil
import pickle
from functools import partial
import argparse
import h5py
from datetime import datetime

import numpy as np
import optax
import haiku as hk
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('root', type=str, help='hdf root directory')
parser.add_argument('species', type=str, help='species name')
parser.add_argument('resolution', type=int, help='model resolution')
parser.add_argument('--obs_weight', help='Weight on the observation term of the loss', default=1.0, type=float)
parser.add_argument('--dist_weight', help='Weight on the distance penalty in the loss', default=1e-3, type=float)
parser.add_argument('--ent_weight', help='Weight on the joint entropy of the model', default=1e-4, type=float)
parser.add_argument('--dist_pow', help='The exponent of the distance penalty', default=0.4, type=float)
parser.add_argument("--dont_normalize", action="store_true", help="don't normalize distance matrix")
parser.add_argument('--learning_rate', help='Learning rate for Adam optimizer', default=0.1, type=float)
parser.add_argument('--training_steps', help='The number of training iterations', default=600, type=int)
parser.add_argument('--rng_seed', help='Random number generator seed', default=17, type=int)
parser.add_argument('--save_pkl', help='Save parameters as pickle file', action='store_true')
args = parser.parse_args()
print(str(args))

hdf_src = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5')
hdf_dst = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km_obs{args.obs_weight}_ent{args.ent_weight}_dist{args.dist_weight}_pow{args.dist_pow}.hdf5')

with open(os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.pkl'), 'rb') as f:
    dtuple, distance_matrices, masked_densities, cells = pickle.load(f)

distance_matrices = [d ** args.dist_pow for d in distance_matrices]
if not args.dont_normalize:
    distance_matrices = [d * (1 / (100**args.dist_pow)) for d in distance_matrices]

# Get the random seed and optimizer
key = hk.PRNGSequence(args.rng_seed)
optimizer = optax.adam(args.learning_rate)

# Instantiate loss function
loss_fn = jit(partial(loss_fn,
                      cells=cells,
                      true_densities=masked_densities, 
                      d_matrices=distance_matrices, 
                      obs_weight=args.obs_weight, 
                      dist_weight=args.dist_weight,
                      ent_weight=args.ent_weight))

# Run Training and get params and losses
params, loss_dict = train_model(loss_fn,
                                optimizer,
                                args.training_steps,
                                cells,
                                dtuple.weeks,
                                key)

if args.save_pkl:
    with open(os.path.join(args.root, f'{args.species}_params_{args.resolution}_obs{args.obs_weight}_ent{args.ent_weight}_dist{args.dist_weight}_pow{args.dist_pow}.pkl'), 'wb') as f:
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

shutil.copyfile(hdf_src, hdf_dst)
file = h5py.File(hdf_dst, 'r+')

margs = file.create_group('marginals')
for i, f in enumerate(flow_amounts):
    margs.create_dataset(f'Week{i+1}_to_{i+2}', data=f)

del file['distances']
    
del file['metadata/birdflow_model_date'] 
file.create_dataset('metadata/birdflow_model_date', data=str(datetime.today()))

hyper = file.create_group("metadata/hyperparameters")
hyper.create_dataset('obs_weight', data=args.obs_weight)
hyper.create_dataset('ent_weight', data=args.ent_weight)
hyper.create_dataset('dist_weight', data=args.dist_weight)
hyper.create_dataset('dist_pow', data=args.dist_pow)
hyper.create_dataset('learning_rate', data=args.learning_rate)
hyper.create_dataset('training_steps', data=args.training_steps)
hyper.create_dataset('rng_seed', data=args.rng_seed)
hyper.create_dataset('normalized', data=not args.dont_normalize)

loss_vals = file.create_group("metadata/loss_values")
loss_vals.create_dataset('total', data=loss_dict['total'])
loss_vals.create_dataset('obs', data=loss_dict['obs'])
loss_vals.create_dataset('dist', data=loss_dict['dist'])
loss_vals.create_dataset('ent', data=loss_dict['ent'])

file.close()