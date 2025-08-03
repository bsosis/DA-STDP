import copy
import os
import time
import multiprocessing as mp
import itertools as it
import numpy as np
from tqdm import tqdm
import einops
from typing import Any, Dict, List, Tuple

import poisson_model
from data_manager import savedata

NUM_PROCS = 8

# ===============================================================================
# Helper functions
# ===============================================================================
def _get_max_T(sim_kwargs: Dict[str, Any]) -> float:
    """
    Compute maximum simulation time max_T to ensure num_steps dopamine events with rate r_dop
    """
    return (sim_kwargs['num_steps']+0.99)/sim_kwargs['r_dop']

def _mp_helper(args: Tuple[Tuple[int, ...], Dict[str, Any]]) -> Tuple[Tuple[int, ...], Any]:
    """
    args should be tuple of (index/indices, kwarg dict)
    inds (and other quantities) is returned with the result to help identify the input
    """
    inds, sim_kwargs = args
    # Set numpy seed to ensure samples are independent
    np.random.seed([int(i) for i in inds] + [os.getpid(), int(10000*time.time()) % 10000])
    res = poisson_model.simulate(**sim_kwargs)
    return inds, res

# ===============================================================================
# Reusable data collection functions
# ===============================================================================

@savedata
def get_weights_over_time(num_samples: int, sim_kwargs: Dict[str, Any], params_to_vary: List[str], 
                          param_vals_list: List[List[Any]]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vary an arbitrary number of parameters over a list of values and run num_samples simulations for each
    num_samples: number of simulations to run for each parameter set
    sim_kwargs: simulation parameters
    params_to_vary: list of parameter names to vary
    param_vals_list: list of lists of parameter values to vary
    Returns:
        weights: array of weights at each time step, shape (num_param_vals, samples, channels, N, steps)
        actions: array of actions at each time step, shape (num_param_vals, samples, steps)
        DA: array of DA at each time step, shape (num_param_vals, samples, steps)
        action_probs: array of action probabilities at each time step, shape (num_param_vals, samples, steps)
    """
    num_param_vals = [len(vals) for vals in param_vals_list]
    num_steps = sim_kwargs['num_steps']
    N = sim_kwargs['N']

    if sim_kwargs['task'] == 'action selection':
        # Two channels
        weights = np.zeros(tuple(num_param_vals + [num_samples, 2, N, num_steps]))
        # Store actions and DA, and action probs
        actions = np.zeros(tuple(num_param_vals + [num_samples, num_steps]), dtype=int)
        DA = np.zeros(tuple(num_param_vals + [num_samples, num_steps]))
        action_probs = np.zeros(tuple(num_param_vals + [num_samples, num_steps]))
    elif sim_kwargs['task'] == 'value estimation':
        # One channel
        weights = np.zeros(tuple(num_param_vals + [num_samples, 1, N, num_steps]))
        # Store actions, DA, and action probs
        actions = np.zeros(tuple(num_param_vals + [num_samples, num_steps]), dtype=int)
        DA = np.zeros(tuple(num_param_vals + [num_samples, num_steps]))
        action_probs = np.zeros(tuple(num_param_vals + [num_samples, num_steps]))
    else:
        raise ValueError(f'Invalid task: {sim_kwargs["task"]}')

    # Generate parameter sets
    param_sets = []
    for inds in it.product(*[range(n) for n in num_param_vals]):
        cur_kwargs = copy.deepcopy(sim_kwargs)
        for param_ind,param_val_ind in enumerate(inds):
            cur_kwargs[params_to_vary[param_ind]] = param_vals_list[param_ind][param_val_ind]
        if 'max_T' not in cur_kwargs:
            max_T = _get_max_T(sim_kwargs)
            cur_kwargs['max_T'] = max_T
        for k in range(num_samples):
            # Needs to be a tuple so we don't use advanced indexing
            param_sets.append((tuple(list(inds)+[k]), cur_kwargs))

    num_runs = len(param_sets)
    with mp.Pool(NUM_PROCS) as pool:
        for inds, res in tqdm(pool.imap_unordered(_mp_helper, param_sets, 
                                                    chunksize=1+int(num_runs/(25*NUM_PROCS))),
                                total=num_runs):
            # res is: (t_arr, y_arr, inds, in_trains, out_trains, dop_train,
            # w_hist, out_rate_hist, action_hist, action_prob_hist, DA_hist)
            w_hist = res[6] # Shape (steps, channels, N)
            weights[inds] = einops.rearrange(w_hist, 's c N -> c N s')
            action_hist = res[8] # Shape (steps,)
            actions[inds] = action_hist
            DA_hist = res[10] # Shape (steps, )
            DA[inds] = DA_hist
            action_prob_hist = res[9] # Shape (steps,)
            action_probs[inds] = action_prob_hist
    return weights, actions, DA, action_probs


@savedata
def get_weights_over_time_multiple(num_samples: int, sim_kwargs: Dict[str, Any], 
                                   param_set_list: List[Tuple[List[str], List[List[Any]]]]
                                   ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Wrap multiple sequential calls to get_weights_over_time
    num_samples: number of simulations to run for each parameter set
    sim_kwargs: simulation parameters
    param_set_list: list of (params_to_vary, param_vals_list) tuples
    Returns list of (weights, actions, DA, action_probs) tuples
    """
    weights_list = [
        get_weights_over_time(num_samples, sim_kwargs, params_to_vary, param_vals_list)
        for params_to_vary, param_vals_list in param_set_list]
    return weights_list
