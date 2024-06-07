import itertools as it
import numpy as np
import scipy.special as scs
import scipy.stats as scst


def get_spike_train(r, T):
    """
    r: rate
    T: max time
    """
    if r == 0:
        return []
    else:
        train = []
        cur_t = np.random.exponential(1/r)
        while cur_t < T:
            train.append(cur_t)
            cur_t += np.random.exponential(1/r)
        return train

def get_expected_action_prob(r1, r2, beta, T_win, err=0.001):
    """
    Get the expected probability of picking action 1
    CDF of Poisson is regularized upper incomplete Gamma, so find n such that (1-err) is counted
    Note, gammainccinv is inverse in wrong variable
    """
    i_n = 10
    while scs.gammaincc(i_n+1, r1*T_win) < 1-err:
        i_n += 5
    j_n = 10
    while scs.gammaincc(j_n+1, r2*T_win) < 1-err:
        j_n += 5
    ij_prod = np.array(list(it.product(range(i_n), range(j_n))))
    moment1 = np.sum(scst.poisson.pmf(ij_prod[:,0], r1*T_win) 
                  * scst.poisson.pmf(ij_prod[:,1], r2*T_win) 
                  * scs.softmax(ij_prod*beta/T_win, axis=1)[:,0])
    return moment1

def filter_in_train(spike_train, task, persistent_activity_rate, r_dop, T_del, T_win):
    """
    For action selection, cut the activity level outside the action selection window
    to persistent_activity_rate (or to zero if it's None) in *both* channels
    (Then the simulation will dynamically drop inputs in unselected channels later)
    """
    if task != 'action selection':
        return spike_train
    else:
        spikes_arr = np.array(spike_train)
        dop_period = 1/r_dop
        filter_prob = 0 if persistent_activity_rate is None else persistent_activity_rate
        spikes_modulo_dop = spikes_arr % dop_period
        # Find the indices of the spikes that are inside the window or that pass filtering
        r = np.random.random(len(spikes_arr))
        # Boolean array
        inds_to_keep = (
            ((spikes_modulo_dop >= dop_period - T_del - T_win) 
             & (spikes_modulo_dop < dop_period - T_del)) # Inside window
            | (r < filter_prob)) # or passes filtering
        return spikes_arr[inds_to_keep].tolist() # Other code assumes this is a list


def simulate(max_T, N, r, Rstar, task, model, alpha, tau, tau_dop, tau_eli, lambda_,
        r_dop, T_del, T_win, eps, beta, w_init,
        DA_mean=None, DA_std=None,
        switch_period=None, same_input_spike_trains=False, persistent_activity_rate=None, 
        single_trace=False, gamma=1,
        store_all=True, **kwargs):
    """
    r should be reshapable to (states, N)
    Rstar should be reshapable to (states, channels)
    channels determined by task
    models one of 'additive', 'multiplicative', 'corticostriatal'
    task is one of 'random', 'reward prediction', 'action selection'
    'action selection' has two channels, others have 1
    states is 1 if switch_period is None, otherwise 2
    If task is 'random' then Rstar is ignored; DA_mean, DA_std should be reshapable to (states,)
    If task is 'action selection' then inputs are silenced during the delay period
    If persistent_activity_rate is given then input in the selected channel is maintained at that rate
    for the action selection model; otherwise both channels go to zero
    Note, with silencing the inputs are still tracked, they just don't impact the post-synaptic neuron
    If single_trace, use a single eligibility trace rather than two,
    and scale negative component by gamma (otherwise gamma is ignored)
    """

    if task == 'random' or task == 'reward prediction':
        channels = 1
    elif task == 'action selection':
        channels = 2
    if switch_period is None:
        states = 1
    else:
        states = 2

    r = np.array(r).reshape(states, N)
    
    if task == 'random':
        DA_mean = np.array(DA_mean).reshape(states)
        DA_std = np.array(DA_std).reshape(states)
    else:
        Rstar = np.array(Rstar).reshape(states, channels)

    # Generate input spike trains
    # Shape (states, channels, N, num_spikes) where num_spikes is variable
    if same_input_spike_trains:
        # Different channels use the same inputs
        # Note, this copies references to the list
        in_trains = [[[filter_in_train(get_spike_train(r[i,k], max_T), 
                                       task, persistent_activity_rate, r_dop, T_del, T_win)
                        for k in range(N)]]*channels 
                        for i in range(states)]
    else:
        in_trains = [[[filter_in_train(get_spike_train(r[i,k], max_T), 
                                       task, persistent_activity_rate, r_dop, T_del, T_win)
                        for k in range(N)] 
                        for _ in range(channels)]
                        for i in range(states)]

    # Convert to a numpy array, padded with 2*max_T at the end (with each having at least one pad)
    max_len = max(max(max(len(in_trains[i][j][k]) for k in range(N)) for j in range(channels)) for i in range(states))
    # Shape (states, channels, N, max_len+1)
    in_trains_arr = np.array([[[in_trains[i][j][k] + [2*max_T]*(max_len - len(in_trains[i][j][k]) + 1) 
                                for k in range(N)] 
                                for j in range(channels)] 
                                for i in range(states)])

    # Make arrays for out trains
    # Note that we use lists here because they can be different lengths
    out_trains = [np.zeros(0) for _ in range(channels)]
    dop_train = np.arange(1/r_dop, max_T, 1/r_dop) # Periodic DA input

    # Precompute some quantities
    len_dop = len(dop_train)
    len_out_trains = [0]*channels

    # Track state at DA release
    w_hist = np.zeros((len_dop, channels, N))
    DA_hist = np.zeros(len_dop)
    if task == 'action selection' or task == 'reward prediction':
        out_rate_hist = np.zeros((len_dop, channels))
    if task == 'action selection':
        action_hist = np.zeros(len_dop, dtype=int)
        action_prob_hist = np.zeros(len_dop)

    # Track which spike is next
    N_range = np.arange(N) # Used for indexing
    next_in_inds = np.zeros((states, channels, N), dtype=int)
    next_out_inds = np.zeros(channels, dtype=int)
    next_dop_ind = 0
    selected_channel = None

    cur_t = 0

    # Track state switching
    state = 0
    next_state_ind = 0
    if switch_period is not None:
        # Only switch states once we enter the spike counting window
        # This correctly selects dop signals at intervals of switch_period
        switch_times = dop_train[switch_period::switch_period] - T_win - T_del
        len_switch = len(switch_times)
    else:
        switch_times = None
        len_switch = 0

    # Main state variables
    t_arr = [0]
    # Coordinates: (Apre, Apost, E_pos, E_neg, DA, w)*channels
    # Apre, E_pos, E_neg, w have length N
    y_arrs = [np.zeros((channels, N + 1 + N + N + 1 + N))]
    # Indices of each variable
    ind_Apre = np.arange(N)
    ind_Apost = np.array([N]) # Note, size-1 array
    ind_E_pos = np.arange(N+1, 2*N+1)
    ind_E_neg = np.arange(2*N+1, 3*N+1)
    ind_DA = np.array([3*N+1]) # Note, size-1 array
    ind_w = np.arange(3*N+2, 4*N+2)

    # Initialize weights
    y_arrs[0][:,ind_w] = w_init # Will broadcast correctly if it's an array


    while cur_t < max_T:
        # Get state
        if switch_period is not None:
            state = next_state_ind % states
            
        # This generates a sequence of indices (state, channel, k, ind) for k=1,...,N, ind the next spike index
        # Gives array of shape (channels, N)
        in_trains_at_next_inds = np.array([in_trains_arr[state, j, N_range, 
                                        next_in_inds[state,j,:]] for j in range(channels)])

        # Find the next spike times
        # Falls back to max_T if no more spikes left
        next_spike_time = min([
                # Next in spike
                np.min(in_trains_at_next_inds),
                # Next DA signal
                dop_train[next_dop_ind] if next_dop_ind < len_dop else max_T,
                # Need to stop at decision time for action selection model
                dop_train[next_dop_ind]-T_del if next_dop_ind < len_dop
                        and task == 'action selection' 
                        and dop_train[next_dop_ind]-T_del > cur_t 
                    else max_T,
                # Next switch time
                switch_times[next_state_ind] if next_state_ind < len_switch else max_T,
                ]
                # Next out spike for each channel
            + [out_trains[j][next_out_inds[j]] if next_out_inds[j] < len_out_trains[j] 
                else max_T for j in range(channels)])

        # Simulate the model until that spike
        dt = next_spike_time - cur_t 
        cur_t = next_spike_time
        if store_all:
            t_arr.append(next_spike_time)
            y_arr = np.copy(y_arrs[-1])
            y_arrs.append(y_arr)
        else:
            # Just overwrite the old array in-place
            # t_arrs not used
            y_arr = y_arrs[-1]

        # Do w first so we can compute it in-place with the old values of the other variables
        if model == 'additive':
            y_arr[:,ind_w] += (lambda_ * (tau_dop*tau_eli/(tau_dop + tau_eli)) 
                                        * (1 - np.exp(-dt*(1/tau_dop + 1/tau_eli)))
                                        * y_arr[:,ind_DA] * (y_arr[:,ind_E_pos] - alpha*y_arr[:,ind_E_neg]))
        # For multiplicative model, see:
        # https://www.wolframalpha.com/input?i=dw%2Fdt+%3D+%28b%2Bc*w%29*e%5E%28-a*t%29%2C+w%280%29+%3D+w_0
        elif model == 'multiplicative':
            a = (1/tau_dop + 1/tau_eli)
            c = -lambda_*y_arr[:,ind_DA]*(y_arr[:,ind_E_pos] + alpha*y_arr[:,ind_E_neg])
            # Deal with division by zero
            denom = y_arr[:,ind_E_pos] + alpha*y_arr[:,ind_E_neg]
            b_over_c = np.divide(-y_arr[:,ind_E_pos], denom, out=np.zeros_like(denom), where=denom!=0)
            y_arr[:,ind_w] = (b_over_c + y_arr[:,ind_w])*np.exp((c/a)*(1 - np.exp(-a*dt))) - b_over_c
        # For corticostriatal model, sign of DA doesn't switch between spikes, so it's similar to multiplicative
        elif model == 'corticostriatal':
            a = (1/tau_dop + 1/tau_eli)
            # Need to do channels separately
            for j in range(channels):
                if y_arr[j,ind_DA][0] >= 0: # Note, y_arr[j,ind_DA] is a size-1 array, so unpack it
                    c = -lambda_*y_arr[j,ind_DA]*(y_arr[j,ind_E_pos] + alpha*y_arr[j,ind_E_neg])
                    denom = y_arr[j,ind_E_pos] + alpha*y_arr[j,ind_E_neg]
                    b_over_c = np.divide(-y_arr[j,ind_E_pos], denom, out=np.zeros_like(denom), where=denom!=0)
                else:
                    c = lambda_*y_arr[j,ind_DA]*(y_arr[j,ind_E_neg] + alpha*y_arr[j,ind_E_pos])
                    denom = y_arr[j,ind_E_neg] + alpha*y_arr[j,ind_E_pos]
                    b_over_c = np.divide(-y_arr[j,ind_E_neg], denom, out=np.zeros_like(denom), where=denom!=0)
                y_arr[j,ind_w] = (b_over_c + y_arr[j,ind_w])*np.exp((c/a)*(1 - np.exp(-a*dt))) - b_over_c

        elif model == 'symmetric':
            # https://www.wolframalpha.com/input?i=dw%28t%29%2Fdt+%3D+k*w*%281-w%29*exp%28-a*t%29%2C+w%280%29%3Dw_0
            a = (1/tau_dop + 1/tau_eli)
            k = lambda_*y_arr[:,ind_DA]*(y_arr[:,ind_E_pos] - alpha*y_arr[:,ind_E_neg])
            y_arr[:,ind_w] = 1/(1 - (1 - 1/(y_arr[:,ind_w] + 1e-8))*np.exp((k/a)*(np.exp(-a*dt) - 1)))

        # Now do other terms
        exp_tau = np.exp(-dt/tau)
        exp_eli = np.exp(-dt/tau_eli)
        y_arr[:,ind_Apre] *= exp_tau
        y_arr[:,ind_Apost] *= exp_tau
        y_arr[:,ind_E_pos] *= exp_eli
        y_arr[:,ind_E_neg] *= exp_eli
        y_arr[:,ind_DA] *= np.exp(-dt/tau_dop)

        # Artifically set w to [0,1] if they've gone a bit outside it
        y_arr[:,ind_w] = np.clip(y_arr[:,ind_w], 0, 1)

        # Track single eligibility trace
        if single_trace:
            # In each channel exactly one of E_pos and E_neg should be nonzero
            eligibility = y_arr[:,ind_E_pos] - y_arr[:,ind_E_neg] # (channels, N)

        # Check in trains for a spike
        # Note, for trains that have run out of spikes the default is 2*max_T so this works
        in_train_spikes = np.abs(in_trains_at_next_inds - cur_t) < 1e-8 # Shape (channels, N)
        next_in_inds[state, in_train_spikes] += 1 # This broadcasts correctly
        # If we're doing action selection and we're not in the action selection window, silence channels
        if (task == 'action selection' and next_dop_ind < len_dop and not
                 (dop_train[next_dop_ind]-T_del-T_win < cur_t <= dop_train[next_dop_ind]-T_del)): # Not in window
            # Only count spikes in selected channels
            # At the beginning of the experiment when selected_channel is None just silence both
            if selected_channel is not None:
                j = selected_channel
                # Increment Apre
                y_arr[j,ind_Apre[in_train_spikes[j]]] += 1
                # Increment E_neg
                if single_trace:
                    eligibility[j,in_train_spikes[j]] -= gamma*y_arr[j,ind_Apost] # Decrement by Apost
                else:
                    y_arr[j,ind_E_neg[in_train_spikes[j]]] += y_arr[j,ind_Apost] # Increment by Apost
                # See if we need to add a post spike
                if np.random.random() < np.dot(y_arr[j,ind_w], in_train_spikes[j])/N:
                    # Note, if eps < 1e-8 this can lead to the spike being processed immediately
                    out_trains[j] = np.append(out_trains[j], cur_t+eps)
                    len_out_trains[j] += 1
        else:
            for j in range(channels):
                # Increment Apre
                y_arr[j,ind_Apre[in_train_spikes[j]]] += 1
                # Increment E_neg
                if single_trace:
                    eligibility[j,in_train_spikes[j]] -= gamma*y_arr[j,ind_Apost] # Decrement by Apost
                else:
                    y_arr[j,ind_E_neg[in_train_spikes[j]]] += y_arr[j,ind_Apost] # Increment by Apost
            # See if we need to add a post spike
            for j in range(channels):
                if np.random.random() < np.dot(y_arr[j,ind_w], in_train_spikes[j])/N:
                    # Note, if eps < 1e-8 this can lead to the spike being processed immediately
                    out_trains[j] = np.append(out_trains[j], cur_t+eps)
                    len_out_trains[j] += 1

        # Check out train for a spike
        for j in range(channels):
            if next_out_inds[j] < len_out_trains[j] and np.abs(out_trains[j][next_out_inds[j]] - cur_t) < 1e-8:
                next_out_inds[j] += 1
                # Increment Apost
                y_arr[j,ind_Apost] += 1
                # Increment E_pos
                if single_trace:
                    eligibility[j,:] += y_arr[j,ind_Apre] # Increment by Apre
                else:
                    y_arr[j,ind_E_pos] += y_arr[j,ind_Apre] # Increment by Apre
        
        # If single_trace, use single eligibility trace to update the separate E_pos and E_neg variables
        if single_trace:
            # Note, both of these are non-negative, and exactly one is nonzero
            y_arr[:,ind_E_pos] = np.maximum(eligibility, 0)
            y_arr[:,ind_E_neg] = np.maximum(-eligibility, 0)

        # Check for decision times
        if (task == 'action selection' and next_dop_ind < len_dop
                and np.abs(dop_train[next_dop_ind] - T_del - cur_t) < 1e-8):
            # Estimate current firing rate
            for j in range(channels):
                # Note, don't subtract T_del because we're calculating this at T_del
                out_rate_hist[next_dop_ind,j] = np.count_nonzero((cur_t - T_win < out_trains[j]) 
                                                                    & (out_trains[j] <= cur_t))/T_win
            # Note, this explicitly assumes channels=2
            p_A1 = scs.softmax(out_rate_hist[next_dop_ind]*beta)[0]
            action_prob_hist[next_dop_ind] = p_A1

            selected_channel = 0 if np.random.random() < p_A1 else 1
            action_hist[next_dop_ind] = selected_channel # 0 or 1


        # Check dop train for a spike
        if next_dop_ind < len_dop and np.abs(dop_train[next_dop_ind] - cur_t) < 1e-8:
            if task == 'random':
                # Normal distribution
                DA = np.random.normal(DA_mean[state], DA_std[state])
                # Note, only one channel
                y_arr[0,ind_DA] += DA
                DA_hist[next_dop_ind] = DA

            elif task == 'reward prediction':
                # Estimate current firing rate
                # Note, only one channel
                out_rate_hist[next_dop_ind,0] = np.count_nonzero((cur_t - T_win - T_del < out_trains[0]) 
                                                                & (out_trains[0] <= cur_t - T_del))/T_win
                DA_hist[next_dop_ind] = Rstar[state, 0] - out_rate_hist[next_dop_ind,0]
                y_arr[0,ind_DA] += DA_hist[next_dop_ind]

            elif task == 'action selection':
                # Action is selected previously, at the delay time
                # Predicted reward is P(A=A1)xR1* + P(A=A2)xR2*, i.e. assume convergence to the true value
                p_A1_pred = get_expected_action_prob(np.dot(y_arr[0,ind_w], r[state])/N, 
                                                    np.dot(y_arr[1,ind_w], r[state])/N,
                                                    beta, T_win)
                R = p_A1_pred*Rstar[state, 0] + (1-p_A1_pred)*Rstar[state, 1]

                cur_Rstar = Rstar[state, selected_channel]

                DA_hist[next_dop_ind] = cur_Rstar - R

                # Update DA in both channels
                y_arr[:,ind_DA] += DA_hist[next_dop_ind]

            # Store current weights
            w_hist[next_dop_ind] = y_arr[:,ind_w]

            # Update dopamine index
            next_dop_ind += 1

        # Check if there's a state switch
        if (switch_period is not None and next_state_ind < len_switch 
                and np.abs(switch_times[next_state_ind] - cur_t) < 1e-8):
            next_state_ind += 1
            # Need to update the indices that haven't been used
            # Argmax returns first true value along the axis
            next_state = next_state_ind % states
            next_in_inds[next_state] = np.argmax(in_trains_arr[next_state] > cur_t, axis=-1)

    inds = (ind_Apre, ind_Apost, ind_E_pos, ind_E_neg, ind_DA, ind_w)
    if store_all:
        # Concatenate at the end for efficiency
        t_arr = np.array(t_arr)
        # Shape (time, channels, variables) -> (channels, variables, time)
        y_arr = np.transpose(y_arrs, (1,2,0))
    else:
        # Don't store
        t_arr, y_arr = None, None

    if task == 'random':
        return (t_arr, y_arr, inds, in_trains, out_trains, dop_train, 
                w_hist, None, None, None, DA_hist)
    elif task == 'reward prediction':
        return (t_arr, y_arr, inds, in_trains, out_trains, dop_train,
                w_hist, out_rate_hist, None, None, DA_hist)
    elif task == 'action selection':
        return (t_arr, y_arr, inds, in_trains, out_trains, dop_train,
                w_hist, out_rate_hist, action_hist, action_prob_hist, DA_hist)



