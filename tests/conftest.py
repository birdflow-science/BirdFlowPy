import jax.numpy as jnp
import pytest

from flow_model_training import Datatuple, mask_input


@pytest.fixture
def tiny_densities():
    """Two cells, two weeks; values chosen so losses are easy to hand-check."""
    pred_densities = [jnp.array([0.5, 0.5]), jnp.array([0.6, 0.4])]
    true_densities = [jnp.array([0.4, 0.6]), jnp.array([0.5, 0.5])]
    return pred_densities, true_densities


@pytest.fixture
def tiny_flow():
    """A single week-to-week flow matrix and matching distance matrix."""
    flows = [jnp.array([[0.2, 0.3], [0.1, 0.4]])]
    d_matrices = [jnp.array([[1.0, 2.0], [3.0, 4.0]])]
    return flows, d_matrices


@pytest.fixture
def small_training_fixture():
    """A small, fixed synthetic dataset shaped like what mask_input expects:
    3 weeks, 4 cells, no masked-out cells, used for golden-value regression
    tests of loss_fn / train_model.
    """
    weeks = 3
    total_cells = 4
    distances = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    masks = jnp.ones((weeks, total_cells), dtype=bool)
    dtuple = Datatuple(weeks, total_cells, distances, masks)

    true_densities = jnp.array([
        [0.40, 0.30, 0.20, 0.10],
        [0.10, 0.40, 0.30, 0.20],
        [0.25, 0.25, 0.25, 0.25],
    ])

    d_matrices, masked_densities = mask_input(true_densities, dtuple)
    cells = [d.shape[0] for d in masked_densities]

    return dtuple, d_matrices, masked_densities, cells
