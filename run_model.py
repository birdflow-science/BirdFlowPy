from flow_model import FlowModel, model_forward, shift_density, gen_samples
from flow_model_training import Datatuple, process_data, gen_d_matrix, loss_fn, train_model
from plots import line_plot, plot_densities

import os
import pickle
import json
import argparse
from functools import partial

import optax
import haiku as hk
import jax.numpy as jnp
from jax import jit


# Arguments
parser = argparse.ArgumentParser(description='Train flow model')
parser.add_argument('bird_code', help='The six digit code which identifies this bird species')
parser.add_argument('data_name', help='Name of the data which the model is trained on')
parser.add_argument('--data_fp', help='Path to the data which the model will train on', default=None)
parser.add_argument('--obs_weight', help='Weight on the observation term of the loss', default=20.0, type=float)
parser.add_argument('--dist_weight', help='Weight on the distance penalty in the loss', default=0.1, type=float)
parser.add_argument('--ent_weight', help='Weight on the joint entropy of the model', default=0.05, type=float)
parser.add_argument('--dist_pow', help='The exponent of the distance penalty', default=1.0, type=float)
parser.add_argument('--learning_rate', help='Learning rate for Adam optimizer', default=0.1, type=float)
parser.add_argument('--training_steps', help='The number of training iterations', default=1000, type=int)
parser.add_argument('--rng_seed', help='Random number generator seed', default=42, type=int)
parser.add_argument('--experiment_name', help='The name of the folder for the results', default=None, type=str)
parser.add_argument('--dtuple_fp', help='Path to dump the dtuple', default=None)
parser.add_argument('--in_gs', help='Whether this is a part of a grid search script', default=False, type=bool)

args = parser.parse_args()

if args.data_fp == None:
    args.data_fp = f'../../data/saved_npy/{args.data_name}.npy'

model_key = f'{args.data_name}_obs{args.obs_weight}_dist{args.dist_weight}_ent{args.ent_weight}_pow{args.dist_pow}'
    
if args.in_gs:
    # Folder to contain the results of the grid_search
    experimental_folder = os.path.join('..', '..', 'results', f'{args.bird_code}_models', f'{args.data_name}_grid_search_{args.experiment_name}')

    # Make Folders to save the model params
    saved_model_path = os.path.join(experimental_folder, 'saved_models', model_key)
    os.makedirs(saved_model_path, exist_ok=True)
    
    # Load the true densities
    with open(os.path.join(experimental_folder, "densities.pkl"), "rb") as f:
        true_densities = pickle.load(f)

    # Load the DataTuple
    with open(os.path.join(experimental_folder, "dtuple.pkl"), "rb") as f:
        dtuple = pickle.load(f)
else:
    # Folder to contain the results of the grid_search
    experimental_folder = os.path.join('..', '..', 'results', f'{args.bird_code}_models', f'{args.data_name}_{args.experiment_name}')

    # Make Folders to save the model params
    saved_model_path = os.path.join(experimental_folder, 'saved_models', model_key)
    os.makedirs(saved_model_path, exist_ok=True)
  
    # Load and process the data
    data_array = jnp.load(args.data_fp)
    true_densities, dtuple = process_data(data_array)

    # Dump the DataTuple
    with open(os.path.join(saved_model_path, "dtuple.pkl"), "wb") as f:
        pickle.dump(dtuple, f)

# Generate the distance Matrix
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

# Dump the final params
with open(os.path.join(saved_model_path, f'params.pkl'), 'wb') as f:
    pickle.dump(params, f)
    
# Dump the losses
with open(os.path.join(saved_model_path, f'losses.json'), 'w') as f:
    json.dump(loss_dict, f)


#print('Generating Samples')
# Get the samples as an array
#samples = gen_samples(args.num_samples, params, dtuple.nan_mask, key)

# Put them in a dataframe and save them
#df = pd.DataFrame(samples, columns=[f'Week {i}' for i in range(1, dtuple.weeks + 1)])
#df.to_csv(os.path.join(sample_path, f'{model_key}_{args.num_samples}.csv'))