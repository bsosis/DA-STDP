import copy
import os
import time
import multiprocessing as mp
import itertools as it
import numpy as np
from tqdm import tqdm
import einops

import poisson_model

NUM_PROCS = 8

# ===============================================================================
# Helper functions
# ===============================================================================
def _get_time_params(sim_kwargs):
    """Given num_steps, other parameters compute r_dop, max_T"""

    # Make sure there's enough time to read the firing rates, wait the delay, then update the weights
    # Time for e^(-t/tau_dop) = 0.005 (~=0.1s for tau_dop=0.02)
    dop_update_time = -sim_kwargs['tau_dop']*np.log(.005) 
    # Also add time for eligibility, etc. to decay so things equilibrate between DA releases
    decay_time = max(-sim_kwargs['tau_eli']*np.log(.005), -sim_kwargs['tau']*np.log(.005))
    r_dop = 1/(dop_update_time + sim_kwargs['T_del'] + sim_kwargs['T_win'] + decay_time)
    max_T = (sim_kwargs['num_steps']+0.99)/r_dop
    return r_dop, max_T

def _get_max_T(sim_kwargs):
    return (sim_kwargs['num_steps']+0.99)/sim_kwargs['r_dop']

def _mp_helper(args):
    """
    args should be tuple of (index/indices, kwarg dict)
    inds (and other quantities) is returned with the result to help identify the input
    """
    inds, sim_kwargs = args
    np.random.seed([int(i) for i in inds] + [os.getpid(), int(10000*time.time()) % 10000])
    res = poisson_model.simulate(**sim_kwargs)
    return inds, res

# ===============================================================================
# Reusable data collection functions
# ===============================================================================


def get_weights_over_time(num_samples, sim_kwargs, params_to_vary, param_vals_list):
    """
    Vary an arbitrary number of parameters
    If task is 'action selection', also return actions and DA
    """
    num_param_vals = [len(vals) for vals in param_vals_list]
    num_steps = sim_kwargs['num_steps']
    N = sim_kwargs['N']

    if sim_kwargs['task'] == 'action selection':
        # Two channels
        weights = np.zeros(tuple(num_param_vals + [num_samples, 2, N, num_steps]))
        # Store actions and DA
        actions = np.zeros(tuple(num_param_vals + [num_samples, num_steps]), dtype=int)
        DA = np.zeros(tuple(num_param_vals + [num_samples, num_steps]))
    else:
        # One channel
        weights = np.zeros(tuple(num_param_vals + [num_samples, 1, N, num_steps]))

    param_sets = []
    for inds in it.product(*[range(n) for n in num_param_vals]):
        cur_kwargs = copy.copy(sim_kwargs)
        cur_kwargs['store_all'] = False
        for param_ind,param_val_ind in enumerate(inds):
            cur_kwargs[params_to_vary[param_ind]] = param_vals_list[param_ind][param_val_ind]
        if 'r_dop' not in cur_kwargs and 'max_T' not in cur_kwargs:
            r_dop, max_T = _get_time_params(sim_kwargs)
            cur_kwargs['r_dop'] = r_dop
        elif 'max_T' not in cur_kwargs:
            max_T = _get_max_T(sim_kwargs)
            cur_kwargs['max_T'] = max_T
        elif 'r_dop' not in cur_kwargs:
            raise ValueError('Must specify r_dop if max_T is specified')
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
            if sim_kwargs['task'] == 'action selection':
                action_hist = res[8] # Shape (steps,)
                actions[inds] = action_hist
                DA_hist = res[10] # Shape (steps, )
                DA[inds] = DA_hist
    if sim_kwargs['task'] == 'action selection':
        return weights, actions, DA
    else:
        return weights

def get_weights_over_time_multiple(num_samples, sim_kwargs, param_set_list):
    """
    Wrap multiple sequential calls to _get_weights_over_time
    param_set_list should be list of (params_to_vary, param_vals_list) tuples
    """
    weights_list = [
        get_weights_over_time(num_samples, sim_kwargs, params_to_vary, param_vals_list)
        for params_to_vary, param_vals_list in param_set_list]
    return weights_list
