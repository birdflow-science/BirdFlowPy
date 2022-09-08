# Bridflow

Birdflow is a modeling framework for estimating individual movement trajectories from population level data. The model takes in eBird Status & Trends abundance estimates and proxy for energetic cost and outputs a time heterogeneous Markov chain which estimates the track distribution of the given species.

## Description

This codebase allows for the gpu-accelerated training and use of Birdflow models in python and gives an example of how to process an abundnce estimate so that it can be used with Birdflow. For a more in depth discussion of the model and experimental results for 11 species in North America, see our preprint [here](https://www.biorxiv.org/content/10.1101/2022.04.12.488057v1).

## Getting Started

### Dependencies

The code is written in python 3.X and depends on the following libraries:
* jax
* haiku
* optax
* scipy

It should be up to date with the latest versions of those libraries

### Executing program

For an example of how to run the code, see the jupyter notebook birdflow_demo.ipynb 

## Help

For any questions or help, contact Miguel Fuentes at mmfuentes@umass.edu

## License

This project is licensed under the MIT License - see the LICENSE file for details
