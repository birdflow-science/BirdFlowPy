from flow_model_training import Datatuple, mask_input

import os
import pickle
import argparse
import h5py

import numpy as np
import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Run an amewoo model')
parser.add_argument('root', type=str, help='hdf root directory')
parser.add_argument('species', type=str, help='species name')
parser.add_argument('resolution', type=int, help='model resolution')

args = parser.parse_args()
print(str(args))

hdf_src = os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.hdf5')

file = h5py.File(hdf_src, 'r')

true_densities = np.asarray(file['distr']).T

weeks = true_densities.shape[0]
total_cells = true_densities.shape[1]

distance_vector = np.asarray(file['distances'])
masks = np.asarray(file['geom']['dynamic_mask']).T.astype(bool)

dtuple = Datatuple(weeks, total_cells, distance_vector, masks)
distance_matrices, masked_densities = mask_input(true_densities, dtuple)
cells = [d.shape[0] for d in masked_densities]

with open(os.path.join(args.root, f'{args.species}_2021_{args.resolution}km.pkl'), 'wb') as f:
    pickle.dump((dtuple, distance_matrices, masked_densities, cells), f)