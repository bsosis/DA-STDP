import copy

import numpy as np
import scipy.special as scs
import einops
import matplotlib.pyplot as plt

import plot_utils
from plot_utils import Plotter, square_subplots, label_subplot, plot_density, add_density_colorbar
from phase_plane_utils import model_funs

import phase_planes_value_estimation
import phase_planes_action_selection


def plot_action_selection_final_weights_delay(weights, delay_vals, models,
                                        save_folder=None, silent=False):
    """
    Plot the weights at the end of the run while varying delay
    weights should have shape
    (num_models, num_persistent_activity, num_delay, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    num_persistent_activity should be 2 (corresponding to None and some nonzero a_sel)
    """

    # Shape (num_models, num_persistent_activity, num_delay, channels)
    weights_mean = einops.reduce(weights, 'm per delay s c 1 t -> m per delay c t', np.mean)
    weights_std = einops.reduce(weights, 'm per delay s c 1 t -> m per delay c t', np.std)

    num_models = len(models)
    linestyles = ['--', '-']

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(ncols=num_models, sharey='row', squeeze=True,
                                figsize=(1+2*(1+num_models), 3), layout='constrained')

        for k in range(num_models):
            for l in range(2):
                # Only look at the last step
                axs[k].errorbar(delay_vals, weights_mean[k,0,:,l,-1], yerr=weights_std[k,0,:,l,-1],
                                label=f'$w^{l+1}$, not sustained' if k==num_models-1 else None,
                                linestyle=linestyles[0], color=plot_utils.colors['weights'][l], capsize=5)
                axs[k].errorbar(delay_vals, weights_mean[k,1,:,l,-1], yerr=weights_std[k,1,:,l,-1],
                                label=f'$w^{l+1}$, sustained' if k==num_models-1 else None,
                                linestyle=linestyles[1], color=plot_utils.colors['weights'][l], capsize=5)

            axs[k].set_title(models[k].title())
            label_subplot(fig, axs[k], k)
            axs[k].set_xlabel(r'$T_{del}$ (seconds)')
        axs[0].set_ylim(0,1.1)
        axs[0].set_ylabel(r'$w$')
        fig.legend(loc='outside right upper')

        plotter.register(fig, f'Action Selection Weight Limits Delay', formats=plot_utils.fig_save_formats)

def plot_action_selection_weights_over_time_contingency_switching(weights, actions, models,
                                    switch_period, param_to_vary=None, param_vals=None, save_folder=None, silent=False):
    """
    Plot weights over time for action selection models
    Plot weights and actions taken
    weights should have shape (num_models, num_param_vals, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    actions should have shape (num_models, num_param_vals, num_samples, num_steps)
    """

    if param_to_vary is None:
        param_vals = [None]
        # Shape (num_models, num_param_vals, channels, num_steps)
        weights_mean = einops.reduce(weights, 'm s c 1 t -> m 1 c t', np.mean)
        weights_std = einops.reduce(weights, 'm s c 1 t -> m 1 c t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm s t -> m 1 t', np.mean)
        actions_std = einops.reduce(actions, 'm s t -> m 1 t', np.std)
    else:
        # Shape (num_models, num_param_vals, channels, num_steps)
        weights_mean = einops.reduce(weights, 'm p s c 1 t -> m p c t', np.mean)
        weights_std = einops.reduce(weights, 'm p s c 1 t -> m p c t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm p s t -> m p t', np.mean)
        actions_std = einops.reduce(actions, 'm p s t -> m p t', np.std)

    num_models = len(models)
    num_steps = weights.shape[-1]

    slow_switching = switch_period != 1
    switch_inds = np.array(range(switch_period, num_steps, switch_period))
    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # stds should be the same since we're just translating/negating
    proportion_A1 = 1-actions_mean

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(2*len(param_vals), num_models, sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*num_models, 1.5+2.5*len(param_vals)), layout='constrained')

        for k in range(num_models):
            for i in range(len(param_vals)):
                for l in range(2):
                    # Plot weights
                    axs[2*i][k].plot(weights_mean[k,i,l,:],
                                    label=r'$w^{'+f'{l+1}'+r'}$', color=plot_utils.colors['weights'][l])
                    axs[2*i][k].fill_between(np.arange(num_steps),
                                            weights_mean[k,i,l,:]+weights_std[k,i,l,:],
                                            weights_mean[k,i,l,:]-weights_std[k,i,l,:],
                                            color=plot_utils.colors['weights'][l], alpha=0.2)
                # Plot actions
                axs[2*i+1][k].plot(proportion_A1[k,i,:], color=plot_utils.colors['actions'], label='Actions', zorder=1)
                axs[2*i+1][k].fill_between(np.arange(num_steps),
                                        proportion_A1[k,i,:]+actions_std[k,i,:],
                                        proportion_A1[k,i,:]-actions_std[k,i,:],
                                        color=plot_utils.colors['actions'], alpha=0.2, zorder=1)

                if slow_switching:
                    # Plot switches
                    for t in switch_inds:
                        axs[2*i][k].axvline(t, linestyle=':', color='black')
                        axs[2*i+1][k].axvline(t, linestyle=':', color='black')

                label_subplot(fig, axs[2*i][k], 2*i*num_models+k)
                label_subplot(fig, axs[2*i+1][k], (2*i+1)*num_models+k)

        for k in range(num_models):
            axs[0][k].set_title(models[k].title())
            axs[-1][k].set_xlabel('Step')
        for i in range(len(param_vals)):
            axs[2*i][0].set_ylabel(r'$w$')
            axs[2*i+1][0].set_ylabel(r'Proportion $A_1$')
            axs[2*i][0].set_ylim(-0.025, 1.025)
            axs[2*i+1][0].set_ylim(-0.025, 1.025)
    
            if len(param_vals) > 1:
                axs[2*i][0].text(x=-0.25, y=1, 
                               s=plot_utils.param_label_map[param_to_vary][:-1] + f'={param_vals[i]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[2*i][0].transAxes) # Set to axis coords
            
        # Add legend to bottom-left of figure with all elements
        # Create dummy lines for legend
        weights1_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][0], linewidth=2, label=r'$w^1$')
        weights2_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][1], linewidth=2, label=r'$w^2$')
        actions_line = plt.Line2D([0], [0], color=plot_utils.colors['actions'], linewidth=2, label='Actions')
        
        fig.legend(handles=[weights1_line, weights2_line, actions_line], 
                  loc='outside lower left', ncol=3)

        if param_to_vary is None:
            plotter.register(fig, f'Action Selection Weights Over Time Contingency Switching', formats=plot_utils.fig_save_formats)
        else:
            plotter.register(fig, f'Action Selection Weights Over Time Contingency Switching {param_to_vary}', formats=plot_utils.fig_save_formats)

def plot_action_selection_weights_over_time_2d(weights, actions, models,
                                    param_to_vary=None, param_vals=None, save_folder=None, silent=False):
    """
    Plot weights over time for action selection models
    Plot weights and actions taken
    weights should have shape (num_models, num_param_vals, num_samples, channels, N, num_steps)
    channels should be 2, N should be 2
    actions should have shape (num_models, num_param_vals, num_samples, num_steps)
    """

    if param_to_vary is None:
        param_vals = [None]
        # Shape (num_models, num_param_vals, channels, num_steps)
        weights_mean = einops.reduce(weights, 'm s c N t -> m 1 c N t', np.mean)
        weights_std = einops.reduce(weights, 'm s c N t -> m 1 c N t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm s t -> m 1 t', np.mean)
        actions_std = einops.reduce(actions, 'm s t -> m 1 t', np.std)
    else:
        # Shape (num_models, num_param_vals, channels, num_steps)
        weights_mean = einops.reduce(weights, 'm p s c N t -> m p c N t', np.mean)
        weights_std = einops.reduce(weights, 'm p s c N t -> m p c N t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm p s t -> m p t', np.mean)
        actions_std = einops.reduce(actions, 'm p s t -> m p t', np.std)

    num_models = len(models)
    num_steps = weights.shape[-1]

    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # stds should be the same since we're just translating/negating
    proportion_A1 = 1-actions_mean

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(2*len(param_vals), num_models, sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*num_models, 1.5+2.5*len(param_vals)), layout='constrained')

        for k in range(num_models):
            for i in range(len(param_vals)):
                for l in range(2):
                    for m in range(2):
                        # Plot weights
                        axs[2*i][k].plot(weights_mean[k,i,l,m,:],
                                        label=r'$w^{'+f'{l+1}'+r'}_{'+f'{m+1}'+r'}$', 
                                        color=plot_utils.colors['weights'][l], linestyle='-' if m == 0 else '--')
                        axs[2*i][k].fill_between(np.arange(num_steps),
                                                weights_mean[k,i,l,m,:]+weights_std[k,i,l,m,:],
                                                weights_mean[k,i,l,m,:]-weights_std[k,i,l,m,:],
                                                color=plot_utils.colors['weights'][l], alpha=0.2)
                # Plot actions
                axs[2*i+1][k].plot(proportion_A1[k,i,:], color=plot_utils.colors['actions'], label='Actions')
                axs[2*i+1][k].fill_between(np.arange(num_steps),
                                        proportion_A1[k,i,:]+actions_std[k,i,:],
                                        proportion_A1[k,i,:]-actions_std[k,i,:],
                                        color=plot_utils.colors['actions'], alpha=0.2)

                label_subplot(fig, axs[2*i][k], 2*i*num_models+k)
                label_subplot(fig, axs[2*i+1][k], (2*i+1)*num_models+k)

        for k in range(num_models):
            axs[0][k].set_title(models[k].title())
            axs[-1][k].set_xlabel('Step')
        for i in range(len(param_vals)):
            axs[2*i][0].set_ylabel(r'$w$')
            axs[2*i+1][0].set_ylabel(r'Proportion $A_1$')
            axs[2*i][0].set_ylim(-0.025, 1.025)
            axs[2*i+1][0].set_ylim(-0.025, 1.025)
    
            if len(param_vals) > 1:
                axs[2*i][0].text(x=-0.25, y=1, 
                               s=plot_utils.param_label_map[param_to_vary][:-1] + f'={param_vals[i]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[2*i][0].transAxes) # Set to axis coords
        
            
        weights11_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][0], linewidth=2, label=r'$w^1_1$', linestyle='-')
        weights12_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][0], linewidth=2, label=r'$w^1_2$', linestyle='--')
        weights21_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][1], linewidth=2, label=r'$w^2_1$', linestyle='--')
        weights22_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][1], linewidth=2, label=r'$w^2_2$', linestyle='-')
        actions_line = plt.Line2D([0], [0], color=plot_utils.colors['actions'], linewidth=2, label='Actions')
        fig.legend(handles=[weights11_line, weights12_line, weights21_line, weights22_line, actions_line], 
                  loc='outside lower left', ncol=5)

        if param_to_vary is None:
            plotter.register(fig, f'Action Selection Weights Over Time 2D', formats=plot_utils.fig_save_formats)
        else:
            plotter.register(fig, f'Action Selection Weights Over Time 2D {param_to_vary}', formats=plot_utils.fig_save_formats)
    



def plot_value_estimation_density_varying_w_init(weights, actions, action_probs, w_init_vals, models, sim_kwargs,
                                         bins=100, save_folder=None, silent=False):
    """
    Plot the density of weights for 2d value estimation
    w should have shape (num_models, num_w_init_vals, num_samples, channels, N, num_steps)
    channels should be 1, N should be 2
    actions should have shape (num_models, num_w_init_vals,num_samples, num_steps)
    """
    # Shape (num_models, num_w_init_vals, num_steps)
    actions_mean = einops.reduce(actions, 'm w s t -> m w t', np.mean)
    actions_std = einops.reduce(actions, 'm w s t -> m w t', np.std)
    # Shape (num_models, num_w_init_vals, num_steps)
    action_probs_mean = einops.reduce(action_probs, 'm w s t -> m w t', np.mean)
    action_probs_std = einops.reduce(action_probs, 'm w s t -> m w t', np.std)

    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # stds should be the same since we're just translating/negating
    proportion_A1 = 1-actions_mean

    num_steps = weights.shape[-1]

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        # Two rows: w and actions
        # Each row needs to share y axis among itself
        fig, axs = plt.subplots(2, len(models), squeeze=False, sharey='row',
                                figsize=(1+2*len(models), 5), layout='constrained')
        for i,model in enumerate(models):
            # Plot w (all initial conditions)
            plot_density(axs[0,i], weights[i,:,:,0,:,:], bins)
            # Plot equilibrium line
            for Rstar in sim_kwargs['Rstar']: # sim_kwargs['Rstar'] should contain two values of Rstar, one for each action
                w1_line = np.linspace(0, 1, 1000)
                w2_line = (2*Rstar - sim_kwargs['r'][0]*w1_line)/sim_kwargs['r'][1]
                inds = [i for i in range(len(w1_line)) if 0<=w1_line[i]<=1 and 0<=w2_line[i]<=1]
                w1_line = w1_line[inds]
                w2_line = w2_line[inds]
                axs[0,i].plot(w1_line, w2_line, color='dimgrey', linestyle='--', zorder=1)

            # Make sure all initial conditions are marked
            for w_init in w_init_vals:
                axs[0,i].scatter(w_init[0], w_init[1], marker='x', color=plot_utils.colors['w_init_marker'], linewidth=1)

            axs[0,i].set_xlim(0, 1)
            axs[0,i].set_ylim(0, 1)
            label_subplot(fig, axs[0,i], i)
            # Only plot the middle w_init value
            # Plot actions
            j = len(w_init_vals)//2
            axs[1][i].plot(proportion_A1[i,j,:], color=plot_utils.colors['actions'], label='Actions', zorder=1)
            axs[1][i].fill_between(np.arange(num_steps),
                                    proportion_A1[i,j,:]+actions_std[i,j,:],
                                    proportion_A1[i,j,:]-actions_std[i,j,:],
                                    color=plot_utils.colors['actions'], alpha=0.2, zorder=1)
            # Plot action probabilities
            axs[1][i].plot(action_probs_mean[i,j,:], color=plot_utils.colors['p'], label=r'$p$', zorder=2)
            axs[1][i].fill_between(np.arange(num_steps),
                                    action_probs_mean[i,j,:]+action_probs_std[i,j,:],
                                    action_probs_mean[i,j,:]-action_probs_std[i,j,:],
                                    color=plot_utils.colors['p'], alpha=0.2, zorder=2)

            axs[1,i].set_ylim(-0.025, 1.025)
            label_subplot(fig, axs[1,i], len(models)+i)

            axs[0,i].set_xlabel(r'$w_1$')
            axs[1,i].set_xlabel('Step')
        
        axs[0,0].set_ylabel(r'$w_2$')
        axs[1,0].set_ylabel(r'$p$')
        
        # Add legend to bottom-left subplot with all three elements
        # Create dummy lines for legend since weights are in a different subplot
        actions_line = plt.Line2D([0], [0], color=plot_utils.colors['actions'], linewidth=2, label='Actions')
        action_probs_line = plt.Line2D([0], [0], color=plot_utils.colors['p'], linewidth=2, label=r'$p$')
        
        fig.legend(handles=[actions_line, action_probs_line], 
                  loc='upper left', bbox_transform=axs[1,-1].transAxes,
                  bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        
        for i,model in enumerate(models):
            axs[0,i].set_title(model.title())
        
        for i in range(len(models)):
            axs[0,i].set_aspect('equal')
        add_density_colorbar(fig, axs[0,:], weights.shape[-1])

        plotter.register(fig, f'Value Estimation Density Varying w_init', formats=plot_utils.fig_save_formats)
    

def plot_value_estimation_phase_planes_density(weights, action_probs, param_to_vary, param_vals, models, sim_kwargs,
                                        bins=100, no_dynamics=False, save_folder=None, silent=False):
    """
    Plot phase planes for 1d value estimation setting, varying model and a parameter
    Plots weights along x-axis, action_probs along y-axis
    Also plot sample trajectories overlaid on the phase planes
    weights should have shape (num_models, len(param_vals), num_samples, channels, N, num_steps)
    N should be 1, channels should be 1
    action_probs should have shape (num_models, len(param_vals), num_samples, num_steps)
    """

    # Combine weights and action_probs
    # Shape (num_models, len(param_vals), num_samples, 1, 2, num_steps)
    combined_data = np.concatenate([weights, action_probs[:,:,:,None,None,:]], axis=-2)

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(len(param_vals), len(models), sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*len(models), 1+2*len(param_vals)))

        for i, model in enumerate(models):
            for j, v in enumerate(param_vals):
                # Plot the phase plane
                cur_kwargs = copy.copy(sim_kwargs)
                cur_kwargs[param_to_vary] = v
                phase_planes_value_estimation.plot_phase_plane(axs[j][i], 
                                              model + ' No Dynamics' if no_dynamics else model, 
                                              cur_kwargs,
                                              points=10, color=plot_utils.colors['vector_field'])
                if not no_dynamics and model in model_funs:
                    phase_planes_value_estimation.plot_center_manifold(axs[j][i], 
                                                                            model,
                                                                            cur_kwargs)

                # Plot densities
                plot_density(axs[j][i], combined_data[i,j,:,0,:,:], bins)

                axs[j][i].set_xlim(0, 1)
                axs[j][i].set_ylim(0, 1)
                label_subplot(fig, axs[j][i], j*len(models)+i)

        for k in range(len(param_vals)):
            axs[k][0].set_ylabel(r'$p$')
            if len(param_vals) > 1:
                axs[k][0].text(x=-0.2, y=1, 
                               s=plot_utils.param_label_map[param_to_vary][:-1] + f'={param_vals[k]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[k][0].transAxes) # Set to axis coords
        for i in range(len(models)):
            axs[-1][i].set_xlabel(r'$w$')
            axs[0][i].set_title(models[i].title())

        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig)
        plt.tight_layout()
        add_density_colorbar(fig, axs, weights.shape[-1], add_legend=not no_dynamics,
                             legend_points=['stable', 'unstable', 'other', 'winit', 'center manifold', 'numerical/analytical', 'solution'])

        plotter.register(fig, f'Value Estimation Phase Planes {param_to_vary} Density', formats=plot_utils.fig_save_formats)

def plot_value_estimation_weights_over_time_contingency_switching(weights, actions, action_probs, models,
                                    switch_period, param_to_vary=None, param_vals=None, save_folder=None, silent=False):
    """
    Plot weights over time for value estimation with contingency switching
    Plot weights and actions taken
    weights should have shape (num_models, num_param_vals, num_samples, channels, N, num_steps)
    channels should be 1, N should be 1
    actions should have shape (num_models, num_param_vals, num_samples, num_steps)
    action_probs should have shape (num_models, num_param_vals, num_samples, num_steps)
    """

    if param_to_vary is None:
        param_vals = [None]
        # Shape (num_models, num_param_vals, num_steps)
        weights_mean = einops.reduce(weights, 'm s 1 1 t -> m 1 t', np.mean)
        weights_std = einops.reduce(weights, 'm s 1 1 t -> m 1 t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm s t -> m 1 t', np.mean)
        actions_std = einops.reduce(actions, 'm s t -> m 1 t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        action_probs_mean = einops.reduce(action_probs, 'm s t -> m 1 t', np.mean)
        action_probs_std = einops.reduce(action_probs, 'm s t -> m 1 t', np.std)
    else:
        # Shape (num_models, num_param_vals, num_steps)
        weights_mean = einops.reduce(weights, 'm p s 1 1 t -> m p t', np.mean)
        weights_std = einops.reduce(weights, 'm p s 1 1 t -> m p t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        actions_mean = einops.reduce(actions, 'm p s t -> m p t', np.mean)
        actions_std = einops.reduce(actions, 'm p s t -> m p t', np.std)
        # Shape (num_models, num_param_vals, num_steps)
        action_probs_mean = einops.reduce(action_probs, 'm p s t -> m p t', np.mean)
        action_probs_std = einops.reduce(action_probs, 'm p s t -> m p t', np.std)
    

    num_models = len(models)
    num_steps = weights.shape[-1]

    slow_switching = switch_period != 1
    switch_inds = np.array(range(switch_period, num_steps, switch_period))
    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # stds should be the same since we're just translating/negating
    proportion_A1 = 1-actions_mean

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(2*len(param_vals), num_models, sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*num_models, 1.5+2.5*len(param_vals)), layout='constrained')

        for k in range(num_models):
            for i in range(len(param_vals)):
                # Plot weights
                axs[2*i][k].plot(weights_mean[k,i,:], color=plot_utils.colors['weights'][0])
                axs[2*i][k].fill_between(np.arange(num_steps),
                                        weights_mean[k,i,:]+weights_std[k,i,:],
                                        weights_mean[k,i,:]-weights_std[k,i,:],
                                        color=plot_utils.colors['weights'][0], alpha=0.2)
                # Plot actions
                axs[2*i+1][k].plot(proportion_A1[k,i,:], color=plot_utils.colors['actions'], label='Actions', zorder=1)
                axs[2*i+1][k].fill_between(np.arange(num_steps),
                                        proportion_A1[k,i,:]+actions_std[k,i,:],
                                        proportion_A1[k,i,:]-actions_std[k,i,:],
                                        color=plot_utils.colors['actions'], alpha=0.2, zorder=1)
                # Plot action probabilities
                axs[2*i+1][k].plot(action_probs_mean[k,i,:], color=plot_utils.colors['p'], label=r'$p$', zorder=2)
                axs[2*i+1][k].fill_between(np.arange(num_steps),
                                        action_probs_mean[k,i,:]+action_probs_std[k,i,:],
                                        action_probs_mean[k,i,:]-action_probs_std[k,i,:],
                                        color=plot_utils.colors['p'], alpha=0.2, zorder=2)

                if slow_switching:
                    # Plot switches
                    for t in switch_inds:
                        axs[2*i][k].axvline(t, linestyle=':', color='black')
                        axs[2*i+1][k].axvline(t, linestyle=':', color='black')

                label_subplot(fig, axs[2*i][k], 2*i*num_models+k)
                label_subplot(fig, axs[2*i+1][k], (2*i+1)*num_models+k)

        for k in range(num_models):
            axs[0][k].set_title(models[k].title())
            axs[-1][k].set_xlabel('Step')
        for i in range(len(param_vals)):
            axs[2*i][0].set_ylabel(r'$w$')
            axs[2*i+1][0].set_ylabel(r'Proportion $A_1$')
            axs[2*i][0].set_ylim(-0.025, 1.025)
            axs[2*i+1][0].set_ylim(-0.025, 1.025)

            if len(param_vals) > 1:
                axs[2*i][0].text(x=-0.25, y=1, 
                               s=plot_utils.param_label_map[param_to_vary][:-1] + f'={param_vals[i]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[2*i][0].transAxes) # Set to axis coords
        
        # Add legend to bottom-left subplot with all three elements
        # Create dummy lines for legend since weights are in a different subplot
        weights_line = plt.Line2D([0], [0], color=plot_utils.colors['weights'][0], linewidth=2, label=r'$w$')
        actions_line = plt.Line2D([0], [0], color=plot_utils.colors['actions'], linewidth=2, label='Actions')
        action_probs_line = plt.Line2D([0], [0], color=plot_utils.colors['p'], linewidth=2, label=r'$p$')
        
        fig.legend(handles=[weights_line, actions_line, action_probs_line], 
                  loc='outside lower left', ncol=3)

        if param_to_vary is None:
            plotter.register(fig, f'Value Estimation Weights Over Time Contingency Switching', formats=plot_utils.fig_save_formats)
        else:
            plotter.register(fig, f'Value Estimation Weights Over Time Contingency Switching {param_to_vary}', formats=plot_utils.fig_save_formats)


def get_predicted_instantaneous_drift_value_estimation(param_to_vary, param_vals, model, sim_kwargs):
    """Get predictions of the averaged model for drift in 1D after a single dopamine signal"""
    assert sim_kwargs['N'] == 1
    predicted_dw = []
    for v in param_vals:
        cur_kwargs = sim_kwargs.copy()
        cur_kwargs[param_to_vary] = v
        Rstar1, Rstar2 = cur_kwargs['Rstar'][0], cur_kwargs['Rstar'][1]
        r = cur_kwargs['r']
        alpha = cur_kwargs['alpha']
        tau = cur_kwargs['tau']
        T_win = cur_kwargs['T_win']
        # Leave out r_dop because we're predicting a single DA release
        consts_w = cur_kwargs['lambda_']*cur_kwargs['tau_eli']*cur_kwargs['tau_dop']/cur_kwargs['N']
        consts_p = cur_kwargs['lambda_bar']*cur_kwargs['beta']*cur_kwargs['tau_dop']
        w = cur_kwargs['w_init']
        p = 0.5 # Initial action probability
        if model == 'corticostriatal':
            dw, dp = phase_planes_value_estimation.corticostriatal_vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, 
                                                                                T_win, consts_w, consts_p)
        else:
            dw, dp = phase_planes_value_estimation.vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, consts_w, consts_p, model)
        predicted_dw.append(dw)
    return np.array(predicted_dw)

def plot_value_estimation_instantaneous_drift(weights, param1, param1_vals,
                                                param2, param2_vals, models, 
                                               sim_kwargs=None, val_for_predictions=None, 
                                               save_folder=None, silent=False):
    """
    Plot instantaneous drift rate for value estimation task
    weights should have shape (models, param1_vals, param2_vals, samples, channels, N, num_steps)
    channels = 1, N = 1, num_steps = 2
    param1 is plot as different lines, param2 is plot along the x-axis
    If val_for_predictions is given, use it as the value of param1 to plot predicted
    instantaneous drift
    """

    dw = weights[...,1] - weights[...,0] # Shape (models, param1_vals, param2_vals, samples, 1, 1)

    dw_mean = einops.reduce(dw, 'm p1 p2 s 1 1 -> m p1 p2', np.mean) # Shape (models, param1_vals, param2_vals)
    dw_std = einops.reduce(dw, 'm p1 p2 s 1 1 -> m p1 p2', np.std) # Shape (models, param1_vals, param2_vals)

    num_models = len(models)

    if sim_kwargs is not None and val_for_predictions is not None:
        predicted_dw = [get_predicted_instantaneous_drift_value_estimation(param2, param2_vals, models[k], 
                                                          {**sim_kwargs, param1: val_for_predictions})
                        for k in range(num_models)]
        predicted_dw = np.array(predicted_dw) # Shape (models, param_vals)
    else:
        predicted_dw = None

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        if param2 == 'w_init':
            fig, axs = plt.subplots(ncols=num_models, squeeze=True,
                                    figsize=(1+2.5*num_models, 3))
        else:
            fig, axs = plt.subplots(ncols=num_models, sharey='row', squeeze=True,
                                    figsize=(1+2*num_models, 3))

        for k in range(num_models):
            for l in range(len(param1_vals)):
                axs[k].errorbar(param2_vals, dw_mean[k,l], yerr=dw_std[k,l],
                               label=(plot_utils.param_label_map[param1][:-1] + '=' + f'{param1_vals[l]}' 
                                      + '$' + plot_utils.param_unit_map[param1]), 
                               capsize=5, color=plot_utils.colors['param_vals'][l])
            axs[k].axhline(0, color='black') # Plot zero line
            if predicted_dw is not None:
                axs[k].plot(param2_vals, predicted_dw[k], color=plot_utils.colors['predicted'], linestyle='--',
                            label='Predicted', zorder=4)

            axs[k].set_title(models[k].title())
            label_subplot(fig, axs[k], k)
            if param2 in plot_utils.param_label_map:
                axs[k].set_xlabel(plot_utils.param_label_map[param2])
            else:
                axs[k].set_xlabel(param2)
            axs[k].set_xlim(0, axs[k].get_xlim()[1])
        axs[0].set_ylabel(r'$\Delta w$')
        if param2 == 'w_init':
            axs[-1].legend(loc='lower left')
        else:
            axs[-1].legend(loc='upper right')
        if param2 == 'T_win' and param2[0] < 0.2:
            # Set ylims to ignore the first entry since it can be extreme
            axs[0].set_ylim(
                1.2*min(np.min(dw_mean[:,:,1:] - dw_std[:,:,1:]), np.min(predicted_dw[:,1:])),
                1.2*max(np.max(dw_mean[:,:,1:] + dw_std[:,:,1:]), np.max(predicted_dw[:,1:]))
            )
        
        plt.tight_layout()

        plotter.register(fig, f'Value Estimation Instantaneous Drift {param1} {param2}', formats=plot_utils.fig_save_formats)

def plot_action_selection_phase_planes_density(weights, param_to_vary, param_vals, models, sim_kwargs,
                                        bins=100, no_dynamics=False, save_folder=None, silent=False):
    """
    Plot phase planes for 1d action selection setting, varying model and a parameter
    Plots w1, w2 along x and y axes
    Also plot sample trajectories overlaid on the phase planes
    weights should have shape (num_models, len(param_vals), num_samples, channels, N, num_steps)
    N should be 1, channels should be 2
    """

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(len(param_vals), len(models), sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*len(models), 1+2*len(param_vals)))

        for i, model in enumerate(models):
            for j, v in enumerate(param_vals):
                # Plot the phase plane
                cur_kwargs = copy.copy(sim_kwargs)
                cur_kwargs[param_to_vary] = v
                phase_planes_action_selection.plot_phase_plane(axs[j][i], 
                                              model + ' No Dynamics' if no_dynamics else model, 
                                              cur_kwargs,
                                              points=10, color=plot_utils.colors['vector_field'])

                # Plot densities
                plot_density(axs[j][i], weights[i,j,:,:,0,:], bins)

                axs[j][i].set_xlim(0, 1)
                axs[j][i].set_ylim(0, 1)
                label_subplot(fig, axs[j][i], j*len(models)+i)

        for k in range(len(param_vals)):
            axs[k][0].set_ylabel(r'$w^2$')
            if len(param_vals) > 1:
                axs[k][0].text(x=-0.2, y=1, 
                               s=plot_utils.param_label_map[param_to_vary][:-1] + f'={param_vals[k]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[k][0].transAxes) # Set to axis coords
        for i in range(len(models)):
            axs[-1][i].set_xlabel(r'$w^1$')
            axs[0][i].set_title(models[i].title())

        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig)
        plt.tight_layout()
        add_density_colorbar(fig, axs, weights.shape[-1], add_legend=not no_dynamics,
                             legend_points=['stable', 'unstable', 'other', 'winit', 'numerical/analytical', 'solution'])

        plotter.register(fig, f'Action Selection Phase Planes {param_to_vary} Density', formats=plot_utils.fig_save_formats)
