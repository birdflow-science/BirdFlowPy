import haiku as hk

from jax.nn import softmax
import jax.numpy as jnp
from jax.random import categorical
from jax.ops import index, index_update


class InitialLoc(hk.Module):
    def __init__(self, cells):
        super().__init__(name='Initial_Params')
        self.cells = cells
        
    
    def __call__(self):
        z0 = hk.get_parameter(
            'z0',
            (self.cells,),
            init=hk.initializers.RandomNormal(),
            dtype = 'float32'
        )
        return softmax(z0)


class FlowBlock(hk.Module):
    def __init__(self, num_cells, week_num=None):
        if week_num:
            name = f'Week_{week_num}'
        else:
            name = 'transition_block'
        super().__init__(name=name)
        self.num_cells = num_cells
        
        
    def __call__(self, last_week):
        z = hk.get_parameter(
            'z',
            (self.num_cells, self.num_cells),
            init=hk.initializers.RandomNormal(),
            dtype = 'float32'
        )
        
        trans_prop = softmax(z, axis=1)
        flow = trans_prop * last_week.reshape(-1, 1)
        return flow
    

class FlowModel(hk.Module):
    def __init__(self, cells, num_weeks, name='Flow_Model'):
        super().__init__(name=name)
        self.num_weeks = num_weeks
        self.cells = cells
        
        
    def __call__(self):
        d0 = InitialLoc(self.cells)()
        d = d0
        flow_amounts = []
        for week in range(self.num_weeks - 1):
            flow = FlowBlock(self.cells, week_num=week + 1)(d)
            flow_amounts.append(flow)
            d = flow.sum(axis=0)
        return (d0, jnp.array(flow_amounts))

def predict(cells, weeks):
    model = FlowModel(cells, weeks)
    return model()
  
model_forward = hk.transform(predict)

def get_prob(param, week, p1, p2):
    logits = param[f'Flow_Model/Week_{week}']['z'][p1, :]
    probs = softmax(logits)
    return probs[p2]
      
def sample_trajectory(rng_seq, flow_params, ipos=None, start=1, end=None):
    if end:
        end = end
    else:
        end = len(flow_params)
            
    if ipos:
        pos = ipos
    else:
        init_p = flow_params['Flow_Model/Initial_Params']['z0']
        pos = categorical(next(rng_seq), init_p)
    
    trajectory = [int(pos)]

    for week in range(start, end):
        trans_p = flow_params[f'Flow_Model/Week_{week}']['z'][pos, :]
        pos = categorical(next(rng_seq), trans_p)
        trajectory.append(int(pos))
    return trajectory

def project_density(flow_params, init_dist, t_start=1, t_end=51):
    d = init_dist
    flow_amounts = []
    for week in range(t_start, t_end):
        z = flow_params[f'Flow_Model/Week_{week}']['z']
        trans_prop = softmax(z, axis=1)
        flow = trans_prop * d.reshape(-1, 1)
        flow_amounts.append(flow)
        d = flow.sum(axis=0)
    return (init_dist, jnp.array(flow_amounts))
  
def gen_shift_list(nan_mask):
    shift = 0
    shift_list = []

    for b in nan_mask:
        if b:
            shift += 1
        else:
            shift_list.append(shift)
    return shift_list
  
def shift_density(pred_density, nan_mask, x_dim, y_dim):
    full_densities = []
    for p in pred_density:
        full_cells = jnp.full((x_dim * y_dim,), jnp.nan)
        full_cells = index_update(full_cells, ~nan_mask, p)
        full_densities.append(full_cells)

    return jnp.array(full_densities)
  
def gen_samples(n, params, nan_mask, key,  ipos=None, start=1, end=None):
    shift_list = gen_shift_list(nan_mask)
    sample_list = []

    for _ in range(n):
        original_sample = sample_trajectory(key, params, ipos=ipos, start=start, end=end)
        shifted_sample = [loc + shift_list[loc] for loc in original_sample]
        sample_list.append(shifted_sample)

    return jnp.array(sample_list)
  
def aggregate_samples(sample_arr, weeks, cells):
    sampled_density = jnp.zeros((weeks, cells))
    for week in weeks:
        weekly_samples = sample_arr[:, week]
        positions, counts = jnp.unique(weekly_samples, return_counts=True)
        for pos, count in zip(positions, counts):
            sampled_density = index_update(sampled_density, index[week, pos], count / n)
    return sampled_density