import os
import copy

import numpy as np
import einops
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D


plt.rcParams.update({
    'text.usetex': True,
    'savefig.dpi': 600,
})

from plottools import Plotter
import phase_planes

subplot_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

param_label_map = {
    'tau_eli': r'$\tau_{eli}$',
    'tau_dop': r'$\tau_{dop}$',
    'tau': r'$\tau$',
    'alpha': r'$\alpha$',
    'gamma': r'$\gamma$',
    'T_win': r'$T_{win}$ (seconds)',
    'w_init': r'$w_{init}$',
}

def register_fig(plotter, fig, name):
    plotter.register(fig, name, formats=['pdf', 'png'])

def square_subplots(fig):
    # https://stackoverflow.com/questions/51474842/python-interplay-between-axissquare-and-set-xlim/51483579#51483579
    ax1 = fig.axes[0]
    rows, cols = ax1.get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw,figh = fig.get_size_inches()

    axw = figw*(r-l)/(cols+(cols-1)*wspace)
    axh = figh*(t-b)/(rows+(rows-1)*hspace)
    axs = min(axw,axh)
    w = (1-axs/figw*(cols+(cols-1)*wspace))/2.
    h = (1-axs/figh*(rows+(rows-1)*hspace))/2.
    fig.subplots_adjust(bottom=h, top=1-h, left=w, right=1-w)

def label_subplot(fig, ax, label_index, scale=1):
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    trans = mtransforms.ScaledTranslation(scale*12.5/72, -scale*7.5/72, #scale*10/72, -scale*5/72, 
                                          fig.dpi_scale_trans)
    ax.text(0.0, scale, subplot_labels[label_index], 
        transform=ax.transAxes + trans,
        fontsize='medium', verticalalignment='top',
        horizontalalignment='center',
        # Params to match the default legend params
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': '0.8', 
              'alpha': 0.5},
        ) 

density_cmap = plt.get_cmap('viridis').reversed()

def plot_density(ax, weights, bins):
    # Adds density plot to an Axes object
    # weights should have shape (num_samples, N, num_steps)
    # N should be 2
    num_steps = weights.shape[-1]
    num_samples = weights.shape[0]
    bin_arr = np.linspace(0,1,bins+1)
    # Use alpha to indicate density, and color to indicate time
    hists_over_time = [np.histogram2d(weights[:,0,k], weights[:,1,k], 
                            bins=bin_arr)[0].T
                        for k in range(num_steps)]

    # Simple averaging
    # This has shape (num_steps, x, y, 4) giving RGBA (with values in [0,1])
    imgs = np.array([density_cmap(t*np.ones_like(h)/num_steps, alpha=h/num_samples)
                    for t,h in enumerate(hists_over_time)])
    # Average the time weighted by samples (alpha)
    alpha_sum = np.sum(imgs[:,:,:,3:], axis=0) # Sum of alpha, shape (x, y, 1)
    img = np.divide(np.sum(imgs[:,:,:,:3]*imgs[:,:,:,3:], axis=0), # Shape (x, y, 3)
                    alpha_sum,
                    out=np.zeros((bins,bins,3)),
                    where=alpha_sum!=0)
    # Normalize
    scale = 2*(num_steps/1000) # This has units of steps
    alpha = np.clip(alpha_sum/scale, 0, 1) # Shape (x, y, 1)
    ax.imshow(np.full((bins, bins, 3), 255, dtype=np.uint8), 
                extent=(0,1,0,1), zorder=-2) # Add white background
    # For some reason using alpha parameter of imshow doesn't work so do this instead
    img = np.concatenate([img, alpha], axis=-1)
    ax.imshow(img, aspect='equal', extent=(0,1,0,1), origin='lower', zorder=-1)

    # Mark the initial point of all the weights
    ax.scatter(weights[0,0,0], weights[0,1,0], marker='x', color='black', linewidth=1)
    # Let the calling function deal with axis labels, bounds, etc.

def add_density_colorbar(fig, axs, max_step, add_legend=False, 
                         legend_points=['stable', 'unstable', 'saddle', 'winit']):
    # Extend the figsize for colorbar
    fig.set_size_inches(fig.get_size_inches()[0]+1, fig.get_size_inches()[1])
    if not add_legend:
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,max_step), cmap=density_cmap), 
                     ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs, 
                     shrink=0.9, label='Step', pad=0.025, fraction=0.1)
    else:
        legend_elements = []
        if 'stable' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor='black', marker='P', 
                                          markersize=12, label='Stable'))
        if 'unstable' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor='red', marker='P', 
                                          markersize=12, label='Unstable'))
        if 'saddle' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor='green', marker='P', 
                                          markersize=12, label='Saddle'))
        if 'winit' in legend_points:
            legend_elements.append(Line2D([0], [0], color='black', marker='x', 
                                          linewidth=0, markersize=10, label=r'$w_{init}$'))
        upper_right_bbox = axs[0,-1].get_position() if isinstance(axs, np.ndarray) else axs.get_position()
        ur_y = upper_right_bbox.y1
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, ur_y))
        
        fig_h = fig.get_size_inches()[1]
        ur_h_in = (1 - ur_y)*fig_h

        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,max_step), cmap=density_cmap), 
                     ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs, 
                     shrink=(fig_h - ur_h_in - 0.2*len(legend_points))/fig_h,
                     label='Step', pad=0.025, fraction=0.1, anchor=(0,0),
                     aspect=20 if isinstance(axs, np.ndarray) and axs.shape[0] == 3 else 10)


def plot_random_DA_weights_over_time(weights_list, params_to_vary, param_vals_list,
                                        models, save_folder=None, silent=False):
    """
    Plot weights over time as parameters are varied
    weights matrices should have shape (models, param_vals, samples, channels, N, steps)
    channels should be 1
    Means/stds will be taken over pooled (samples, N)
    """

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        # Plot with fill_between in subplots
        fig, axs = plt.subplots(len(params_to_vary), len(models), sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*len(models), 1+2*len(params_to_vary)))

        for i,model in enumerate(models):
            for j,param in enumerate(params_to_vary):
                # These have shape (models, param_vals, steps)
                weights_mean = einops.reduce(weights_list[j], 'm p s 1 N t -> m p t', np.mean)
                weights_std = einops.reduce(weights_list[j], 'm p s 1 N t -> m p t', np.std)
                for k, v in enumerate(param_vals_list[j]):
                    axs[j][i].plot(weights_mean[i,k,:], label=param_label_map[param][:-1] + f'={v}$', 
                                    color=f'C{k}')
                    axs[j][i].fill_between(np.arange(len(weights_mean[i,k,:])), 
                                    weights_mean[i,k,:]+weights_std[i,k,:],
                                    weights_mean[i,k,:]-weights_std[i,k,:],
                                    color=f'C{k}', alpha=0.2)
                if j == 0:
                    axs[0][i].set_title(model.title())
                label_subplot(fig, axs[j][i], i+j*len(models))

        for j in range(len(params_to_vary)):
            axs[j][0].set_ylabel(r'$w$')
            axs[j][0].set_ylim(0,1)
            axs[j][-1].legend()
        for i in range(len(models)):
            axs[-1][i].set_xlabel('Step')

        plt.tight_layout()
        register_fig(plotter, fig, 'Weights over time')


def plot_reward_prediction_phase_planes_density(weights, param_to_vary, param_vals, models, sim_kwargs,
                                        bins=100, no_dynamics=False, save_folder=None, silent=False):
    """
    Plot phase planes varying model and a parameter
    Also plot sample trajectories overlaid on the phase planes
    weights should have shape (num_models, len(param_vals), num_samples, channels, N, num_steps)
    N should be 2, channels should be 1
    Note, does *not* add a label for the parameter
    """

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(len(param_vals), len(models), sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*len(models), 1+2*len(param_vals)))

        for i, model in enumerate(models):
            for j, v in enumerate(param_vals):
                # Plot the phase plane
                cur_kwargs = copy.copy(sim_kwargs)
                cur_kwargs[param_to_vary] = v
                phase_planes.plot_phase_plane(axs[j][i], 
                                              model + ' No Dynamics' if no_dynamics else model, 
                                              cur_kwargs['alpha'], cur_kwargs['tau'],
                                              cur_kwargs['r'], cur_kwargs['Rstar'],
                                              T_win=cur_kwargs['T_win'], points=10, color='dimgray')

                # Plot densities
                plot_density(axs[j][i], weights[i,j,:,0,:,:], bins)

                axs[j][i].set_xlim(0, 1)
                axs[j][i].set_ylim(0, 1)
                label_subplot(fig, axs[j][i], j*len(models)+i)

        for k in range(len(param_vals)):
            axs[k][0].set_ylabel(r'$w_2$')
            if len(param_vals) > 1:
                axs[k][0].text(x=-0.2, y=1, 
                               s=param_label_map[param_to_vary][:-1] + f'={param_vals[k]}$',
                               horizontalalignment='right', verticalalignment='center_baseline',
                               transform=axs[k][0].transAxes) # Set to axis coords
        for i in range(len(models)):
            axs[-1][i].set_xlabel(r'$w_1$')
            axs[0][i].set_title(models[i].title())

        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig)
        plt.tight_layout()
        add_density_colorbar(fig, axs, weights.shape[-1], add_legend=not no_dynamics,
                             legend_points=['stable', 'unstable', 'winit'])

        register_fig(plotter, fig, f'Reward Prediction Phase Planes {param_to_vary} Density')


def plot_reward_prediction_phase_planes_task_switching_density(weights, 
                                        Rstar_vals, models, sim_kwargs,
                                        bins=100, save_folder=None, silent=False):
    """
    Plot phase planes varying model, R*, and whether switching is fast or slow
    Also plot density overlaid on the phase planes
    weights should have shape:
    (num_models, len(Rstar_vals), switch period vals, num_samples, channels, N, num_steps)
    N should be 2, channels should be 1, switch period vals should be 2
    Plots a separate figure for each R* set, varying whether switching is slow or fast within the figure
    """
    
    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        # Plot a combined figure
        fig, axs = plt.subplots(2, len(models)*len(Rstar_vals), sharex='col', sharey='row', squeeze=False,
                                figsize=(1.9*len(models)*len(Rstar_vals), 4))

        for l,Rstar in enumerate(Rstar_vals):
            for i, model in enumerate(models):
                for j in range(2): # Slow, Fast
                    col = i + l*len(models)
                    if j == 0: # Slow
                        phase_planes.plot_phase_plane_slow_switching(axs[j][col], model, 
                                sim_kwargs['alpha'], sim_kwargs['tau'], sim_kwargs['r'], Rstar,
                                T_win=sim_kwargs['T_win'])
                    else: # Fast
                        phase_planes.plot_phase_plane_fast_switching(axs[j][col], model, 
                                sim_kwargs['alpha'], sim_kwargs['tau'], sim_kwargs['r'], Rstar,
                                T_win=sim_kwargs['T_win'], points=10, color='dimgray')
                    # Plot densities
                    plot_density(axs[j][col], weights[i,l,j,:,0,:,:], bins)

                    axs[j][col].set_xlim(0, 1)
                    axs[j][col].set_ylim(0, 1)
                    label_subplot(fig, axs[j][col], j*len(models)*len(Rstar_vals)+col)

        for k in range(2):
            axs[k][0].set_ylabel(r'$w_2$')
        axs[0][0].text(x=-0.5, y=0, 
                        s='Slow\nswitching',
                        rotation=90,
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=axs[0][0].transAxes) # Set to axis coords
        axs[1][0].text(x=-0.5, y=0, 
                        s='Fast\nswitching',
                        rotation=90,
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=axs[1][0].transAxes) # Set to axis coords
        for i in range(len(models)*len(Rstar_vals)):
            axs[0][i].set_title(models[i%len(models)].title())
            axs[-1][i].set_xlabel(r'$w_1$')

        axs[0][0].text(x=0.5, y=1.2, 
                        fontsize=axs[0][0].title.get_fontsize(),
                        s='With intersection',
                        horizontalalignment='center', verticalalignment='bottom',
                        transform=axs[0][0].transAxes) # Set to axis coords
        axs[0][2].text(x=0.5, y=1.2, 
                        fontsize=axs[0][2].title.get_fontsize(),
                        s='Without intersection',
                        horizontalalignment='center', verticalalignment='bottom',
                        transform=axs[0][2].transAxes) # Set to axis coords

        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig)
        plt.tight_layout()
        add_density_colorbar(fig, axs, weights.shape[-1], add_legend=True,
                             legend_points=['stable', 'unstable', 'saddle', 'winit'])

        register_fig(plotter, fig, f'Reward Prediction Phase Planes Density Rstar')



def plot_action_selection_weights_over_time(weights, actions, models,
                                    save_folder=None, silent=False):
    """
    Plot weights over time for action selection models
    Plot weights and actions taken
    weights should have shape (num_models, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    actions should have shape (num_models, num_samples, num_steps)
    """

    # Shape (num_models, channels, num_steps)
    weights_mean = einops.reduce(weights, 'm s c 1 t -> m c t', np.mean)
    weights_std = einops.reduce(weights, 'm s c 1 t -> m c t', np.std)
    # Shape (num_models, num_steps)
    actions_mean = einops.reduce(actions, 'm s t -> m t', np.mean)
    actions_std = einops.reduce(actions, 'm s t -> m t', np.std)

    num_models = len(models)
    num_steps = weights.shape[-1]

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(2, num_models, sharex='col', sharey='row', squeeze=False,
                                )

        for k in range(num_models):
            for l in range(2):
                # Plot weights
                axs[0][k].plot(weights_mean[k,l,:],
                                label=r'$w^{'+f'{l+1}'+r'}$', color=f'C{l}')
                axs[0][k].fill_between(np.arange(num_steps),
                                        weights_mean[k,l,:]+weights_std[k,l,:],
                                        weights_mean[k,l,:]-weights_std[k,l,:],
                                        color=f'C{l}', alpha=0.2)
            # Plot actions
            # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
            axs[1][k].plot(1-actions_mean[k,:], color='C0')
            axs[1][k].fill_between(np.arange(num_steps),
                                    (1-actions_mean[k,:])+actions_std[k,:],
                                    (1-actions_mean[k,:])-actions_std[k,:],
                                    color='C0', alpha=0.2)

            axs[0][k].set_title(models[k].title())
            label_subplot(fig, axs[0][k], k)
            label_subplot(fig, axs[1][k], num_models+k)

            if k == 0:
                axs[0][k].set_ylim(0, 1)
                axs[0][k].set_ylabel(r'$w$')
                axs[1][k].set_ylim(0, 1)
                axs[1][k].set_ylabel(r'Proportion correct')
            axs[1][k].set_xlabel('Step')
        axs[0][-1].legend()

        plt.tight_layout()

        register_fig(plotter, fig, f'Action Selection Weights Over Time')
    
    # Save data at last time point
    with open(os.path.join(save_folder, 'last_step.txt'), 'w') as f:
        for i, model in enumerate(models):
            f.write(f'{model} model:\n')
            f.write(f'\tw^1: {weights_mean[i,0,-1]:.3f} +/- {weights_std[i,0,-1]:.3f}\n')
            f.write(f'\tw^2: {weights_mean[i,1,-1]:.3f} +/- {weights_std[i,1,-1]:.3f}\n')
            f.write(f'\tP(correct): {1-actions_mean[i,-1]:.3f} +/- {actions_std[i,-1]:.3f}\n')


def plot_action_selection_weight_limits_delay(weights, delay_vals, models,
                                        save_folder=None, silent=False):
    """
    Plot the weights at the end of the run while varying delay
    weights should have shape
    (num_models, num_persistent_activity, num_delay, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    num_persistent_activity should be 2 (corresponding to None and some nonzero persistent_activity_rate)
    """

    # Shape (num_models, num_persistent_activity, num_delay, channels)
    weights_mean = einops.reduce(weights, 'm per delay s c 1 t -> m per delay c t', np.mean)
    weights_std = einops.reduce(weights, 'm per delay s c 1 t -> m per delay c t', np.std)

    num_models = len(models)
    linestyles = ['-', '--']

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(ncols=num_models, sharey='row', squeeze=True,
                                figsize=(1+2*num_models, 3))

        for k in range(num_models):
            for l in range(2):
                # Only look at the last step
                axs[k].errorbar(delay_vals, weights_mean[k,0,:,l,-1], yerr=weights_std[k,0,:,l,-1],
                                label=f'$w^{l+1}$, not sustained',
                                linestyle=linestyles[0], color=f'C{l}', capsize=5)
                axs[k].errorbar(delay_vals, weights_mean[k,1,:,l,-1], yerr=weights_std[k,1,:,l,-1],
                                label=f'$w^{l+1}$, sustained',
                                linestyle=linestyles[1], color=f'C{l}', capsize=5)

            axs[k].set_title(models[k].title())
            label_subplot(fig, axs[k], k)
            axs[k].set_xlabel(r'$T_{del}$ (seconds)')
        axs[0].set_ylim(0,1.2) # Use 1.2 because some curves are right at 1
        axs[0].set_ylabel(r'$w$')
        axs[-1].legend(loc='upper right')

        plt.tight_layout()

        register_fig(plotter, fig, f'Action Selection Weight Limits Delay')
    
    # Save data at last time point
    with open(os.path.join(save_folder, 'last_step.txt'), 'w') as f:
        f.write(f'Delay vals: {delay_vals}\n')
        f.write(f'Not sustained:\n')
        for i, model in enumerate(models):
            f.write(f'\t{model} model:\n')
            f.write(f'\t\tw^1: {weights_mean[i,0,:,0,-1]} +/- {weights_std[i,0,:,0,-1]}\n')
            f.write(f'\t\tw^2: {weights_mean[i,0,:,1,-1]} +/- {weights_std[i,0,:,1,-1]}\n')
        f.write(f'Sustained:\n')
        for i, model in enumerate(models):
            f.write(f'\t{model} model:\n')
            f.write(f'\t\tw^1: {weights_mean[i,1,:,0,-1]} +/- {weights_std[i,1,:,0,-1]}\n')
            f.write(f'\t\tw^2: {weights_mean[i,1,:,1,-1]} +/- {weights_std[i,1,:,1,-1]}\n')


def plot_action_selection_weights_over_time_contingency_switching(weights, actions, models,
                                    switch_period, save_folder=None, silent=False):
    """
    Plot weights over time for action selection models
    Plot weights and actions taken
    weights should have shape (num_models, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    actions should have shape (num_models, num_samples, num_steps)
    """

    # Shape (num_models, channels, num_steps)
    weights_mean = einops.reduce(weights, 'm s c 1 t -> m c t', np.mean)
    weights_std = einops.reduce(weights, 'm s c 1 t -> m c t', np.std)
    # Shape (num_models, num_steps)
    actions_mean = einops.reduce(actions, 'm s t -> m t', np.mean)
    actions_std = einops.reduce(actions, 'm s t -> m t', np.std)

    num_models = len(models)
    num_steps = weights.shape[-1]

    slow_switching = switch_period != 1
    switch_inds = np.array(range(switch_period, num_steps, switch_period))
    states = (np.arange(num_steps)//switch_period) % 2
    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # P(correct) = P(A1)*(1-state) + (1-P(A1))*state (since state is 0, 1 for actions 1, 2)
    # stds should be the same since we're just translating/negating
    prob_A1 = 1-actions_mean
    prob_correct = prob_A1*(1 - states) + (1 - prob_A1)*states

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(2, num_models, sharex='col', sharey='row', squeeze=False,
                                figsize=(1+2*num_models, 4))

        for k in range(num_models):
            for l in range(2):
                # Plot weights
                axs[0][k].plot(weights_mean[k,l,:],
                                label=r'$w^{'+f'{l+1}'+r'}$', color=f'C{l}')
                axs[0][k].fill_between(np.arange(num_steps),
                                        weights_mean[k,l,:]+weights_std[k,l,:],
                                        weights_mean[k,l,:]-weights_std[k,l,:],
                                        color=f'C{l}', alpha=0.2)
            # Plot actions
            axs[1][k].plot(prob_correct[k,:], color='C0')
            axs[1][k].fill_between(np.arange(num_steps),
                                    prob_correct[k,:]+actions_std[k,:],
                                    prob_correct[k,:]-actions_std[k,:],
                                    color='C0', alpha=0.2)
            if slow_switching:
                # Plot switches
                for t in switch_inds:
                    axs[0][k].axvline(t, linestyle=':', color='black')
                    axs[1][k].axvline(t, linestyle=':', color='black')

            axs[0][k].set_title(models[k].title())
            label_subplot(fig, axs[0][k], k)
            label_subplot(fig, axs[1][k], num_models+k)

            if k == 0:
                axs[0][k].set_ylim(0, 1)
                axs[0][k].set_ylabel(r'$w$')
                axs[1][k].set_ylim(0, 1)
                axs[1][k].set_ylabel(r'Proportion correct')
            axs[1][k].set_xlabel('Step')
        axs[0][-1].legend(loc='upper right')

        plt.tight_layout()

        register_fig(plotter, fig, f'Action Selection Weights Over Time Contingency Switching')
    
    # Save data at last time point
    with open(os.path.join(save_folder, 'last_step.txt'), 'w') as f:
        for i, model in enumerate(models):
            f.write(f'{model} model:\n')
            f.write(f'\tw^1: {weights_mean[i,0,-1]:.3f} +/- {weights_std[i,0,-1]:.3f}\n')
            f.write(f'\tw^2: {weights_mean[i,1,-1]:.3f} +/- {weights_std[i,1,-1]:.3f}\n')
            f.write(f'\tP(correct): {prob_correct[i,-1]:.3f} +/- {actions_std[i,-1]:.3f}\n')

def plot_action_selection_weights_over_time_contingency_switching_one_model(weights, actions,
                                    switch_period, save_folder=None, silent=False):
    """
    Plot weights over time for action selection models
    Plot weights and actions taken
    weights should have shape (1, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    actions should have shape (1, num_samples, num_steps)
    """

    # Shape (num_models, channels, num_steps)
    weights_mean = einops.reduce(weights, '1 s c 1 t -> c t', np.mean)
    weights_std = einops.reduce(weights, '1 s c 1 t -> c t', np.std)
    # Shape (num_models, num_steps)
    actions_mean = einops.reduce(actions, '1 s t -> t', np.mean)
    actions_std = einops.reduce(actions, '1 s t -> t', np.std)

    num_steps = weights.shape[-1]

    slow_switching = switch_period != 1
    switch_inds = np.array(range(switch_period, num_steps, switch_period))
    states = (np.arange(num_steps)//switch_period) % 2
    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # P(correct) = P(A1)*(1-state) + (1-P(A1))*state (since state is 0, 1 for actions 1, 2)
    # stds should be the same since we're just translating/negating
    prob_A1 = 1-actions_mean
    prob_correct = prob_A1*(1 - states) + (1 - prob_A1)*states

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(1, 2, squeeze=True, #)
                                figsize=(6, 3))

        for l in range(2):
            # Plot weights
            axs[0].plot(weights_mean[l,:],
                            label=r'$w^{'+f'{l+1}'+r'}$', color=f'C{l}')
            axs[0].fill_between(np.arange(num_steps),
                                    weights_mean[l,:]+weights_std[l,:],
                                    weights_mean[l,:]-weights_std[l,:],
                                    color=f'C{l}', alpha=0.2)
        # Plot actions
        axs[1].plot(prob_correct, color='C0')
        axs[1].fill_between(np.arange(num_steps),
                                prob_correct+actions_std,
                                prob_correct-actions_std,
                                color='C0', alpha=0.2)
        if slow_switching:
            # Plot switches
            for t in switch_inds:
                axs[0].axvline(t, linestyle=':', color='black')
                axs[1].axvline(t, linestyle=':', color='black')

        label_subplot(fig, axs[0], 0)
        label_subplot(fig, axs[1], 1)

        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel(r'$w$')
        axs[0].set_xlabel('Step')
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel(r'Proportion correct')
        axs[1].set_xlabel('Step')
        axs[0].legend(loc='upper right')

        plt.tight_layout()

        register_fig(plotter, fig, f'Action Selection Weights Over Time Contingency Switching')
    
    # Save data at last time point
    with open(os.path.join(save_folder, 'last_step.txt'), 'w') as f:
        f.write(f'w^1: {weights_mean[0,-1]:.3f} +/- {weights_std[0,-1]:.3f}\n')
        f.write(f'w^2: {weights_mean[1,-1]:.3f} +/- {weights_std[1,-1]:.3f}\n')
        f.write(f'P(correct): {prob_correct[-1]:.3f} +/- {actions_std[-1]:.3f}\n')

def plot_action_selection_task_switching_density(weights, actions, models, sim_kwargs,
                                         bins=100, save_folder=None, silent=False):
    """
    Plot the density of weights for task switching
    w should have shape (num_models, num_samples, channels, N, num_steps)
    channels should be 2, N should be 2
    actions should have shape (num_models, num_samples, num_steps)
    r_vals should have shape (states, N)
    Rstar_vals should have shape (states, channels)
    states should be 2
    """
    
    slow_switching = sim_kwargs['switch_period'] != 1
    # Shape (num_models, channels, N, num_steps)
    weights_mean = einops.reduce(weights, 'm s c N t -> m c N t', np.mean)
    weights_std = einops.reduce(weights, 'm s c N t -> m c N t', np.std)
    # Shape (num_models, num_steps)
    actions_mean = einops.reduce(actions, 'm s t -> m t', np.mean)
    actions_std = einops.reduce(actions, 'm s t -> m t', np.std)

    num_steps = weights.shape[-1]
    switch_inds = np.array(range(sim_kwargs['switch_period'], num_steps, sim_kwargs['switch_period']))
    states = (np.arange(num_steps)//sim_kwargs['switch_period']) % 2
    # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
    # P(correct) = P(A1)*(1-state) + (1-P(A1))*state (since state is 0, 1 for actions 1, 2)
    # stds should be the same since we're just translating/negating
    prob_A1 = 1-actions_mean
    prob_correct = prob_A1*(1 - states) + (1 - prob_A1)*states

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        # Three rows: w^1, w^2, and actions
        # First two rows need to share x and y axes, third row needs to share y axis among itself
        # https://stackoverflow.com/questions/42973223/how-to-share-x-axes-of-two-subplots-after-they-have-been-created
        fig, axs = plt.subplots(3, len(models), squeeze=False,
                                figsize=(1+2*len(models), 7))
        for i,model in enumerate(models):
            # Plot w^1 (channel 0)
            plot_density(axs[0,i], weights[i,:,0,:,:], bins)
            axs[0,i].set_xlim(0, 1)
            axs[0,i].set_ylim(0, 1)
            label_subplot(fig, axs[0,i], i)
            # Plot w^2 (channel 1)
            plot_density(axs[1,i], weights[i,:,1,:,:], bins)
            axs[1,i].set_xlim(0, 1)
            axs[1,i].set_ylim(0, 1)
            label_subplot(fig, axs[1,i], len(models)+i)
            # Plot actions
            axs[2][i].plot(prob_correct[i,:], color='C0')
            axs[2][i].fill_between(np.arange(num_steps),
                                    prob_correct[i,:]+actions_std[i,:],
                                    prob_correct[i,:]-actions_std[i,:],
                                    color='C0', alpha=0.2)
            if slow_switching:
                # Plot switches
                for t in switch_inds:
                    axs[2][i].axvline(t, linestyle=':', color='black')
            axs[2,i].set_ylim(0, 1)
            label_subplot(fig, axs[2,i], 2*len(models)+i)

            if i != 0:
                axs[0,i].set_yticklabels([]) # Hide y tick labels
                axs[1,i].set_yticklabels([]) # Hide y tick labels
                axs[2,i].set_yticklabels([]) # Hide y tick labels

            axs[0,i].set_xlabel(r'$w_1^1$')
            axs[1,i].set_xlabel(r'$w_2^1$')
            axs[2,i].set_xlabel('Step')
        
        axs[0,0].set_ylabel(r'$w_2^1$')
        axs[1,0].set_ylabel(r'$w_2^2$')
        axs[2,0].set_ylabel(r'Proportion correct')

        # Link axes
        for k in range(3):
            axs[k,0].get_shared_y_axes().join(axs[k,0], *axs[k,1:]) # Link whole row
            axs[0,k].get_shared_x_axes().join(axs[0,k], axs[1,k]) # Link first two in column
        
        for i,model in enumerate(models):
            axs[0,i].set_title(model.title())
        
        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig) # Note, not sure action plots should be square but whatever
        plt.tight_layout()
        add_density_colorbar(fig, axs[:2,:], weights.shape[-1])

        register_fig(plotter, fig, f'Action Selection Task Switching Density {"Slow" if slow_switching else "Fast"}')
    
    # Save data at last time point
    with open(os.path.join(save_folder, 'last_step.txt'), 'w') as f:
        for i, model in enumerate(models):
            f.write(f'{model} model:\n')
            f.write(f'\tw^1: {weights_mean[i,0,:,-1]} +/- {weights_std[i,0,:,-1]}\n')
            f.write(f'\tw^2: {weights_mean[i,1,:,-1]} +/- {weights_std[i,1,:,-1]}\n')
            f.write(f'\tP(correct): {prob_correct[i,-1]:.3f} +/- {actions_std[i,-1]:.3f}\n')


def plot_single_model_all_settings(model, weights_random_DA_winit, winit_vals, 
                             weights_reward_prediction, sim_kwargs_reward_prediction,
                             weights_action_selection, actions,
                             bins=100, save_folder=None, silent=False):
    """
    Plot basic behavior of a single model in all three settings
    weights_random_DA_winit should have shape (winit_vals, samples, channels, N, steps)
    channels should be 1; samples and N will be pooled
    weights_reward_prediction should have shape (1, num_samples, channels, N, num_steps)
    channels should be 1, N should be 2
    weights_action_selection should have shape (1, num_samples, channels, N, num_steps)
    channels should be 2, N should be 1
    """
    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        fig, axs = plt.subplots(1, 3,
                                figsize=(7.75, 2.5))
        # Plot random DA winit
        weights_winit_mean = einops.reduce(weights_random_DA_winit, 'p s 1 N t -> p t', np.mean)
        weights_winit_std = einops.reduce(weights_random_DA_winit, 'p s 1 N t -> p t', np.std)
        for j, v in enumerate(winit_vals):
            axs[0].plot(weights_winit_mean[j,:], label=r'$w_{init}=' + f'{v:.2f}$', color=f'C{j}')
            axs[0].fill_between(np.arange(len(weights_winit_mean[j,:])), 
                                weights_winit_mean[j,:]+weights_winit_std[j,:],
                                weights_winit_mean[j,:]-weights_winit_std[j,:],
                                color=f'C{j}', alpha=0.2)
        axs[0].set_title('Random Dopamine')
        label_subplot(fig, axs[0], 0)
        axs[0].set_xlabel(r'Step')
        axs[0].set_ylabel(r'$w$')
        axs[0].set_ylim(0,1)
        axs[0].legend()

        # Plot reward prediction
        phase_planes.plot_phase_plane(axs[1], model,
                                      sim_kwargs_reward_prediction['alpha'], 
                                      sim_kwargs_reward_prediction['tau'],
                                      sim_kwargs_reward_prediction['r'], 
                                      sim_kwargs_reward_prediction['Rstar'],
                                      T_win=sim_kwargs_reward_prediction['T_win'], 
                                      points=10, color='dimgray')
        plot_density(axs[1], weights_reward_prediction[0,:,0,:,:], bins)
        axs[1].set_title('Reward Prediction')
        label_subplot(fig, axs[1], 1)
        axs[1].set_xlabel(r'$w_1$')
        axs[1].set_ylabel(r'$w_2$')
        axs[1].set_xlim(0, 1)
        axs[1].set_ylim(0, 1)

        # Plot action selection
        # Shape (num_models, channels, num_steps)
        weights_mean = einops.reduce(weights_action_selection, '1 s c 1 t -> c t', np.mean)
        weights_std = einops.reduce(weights_action_selection, '1 s c 1 t -> c t', np.std)
        num_steps = weights_action_selection.shape[-1]
        for l in range(2):
            # Plot weights
            axs[2].plot(weights_mean[l,:],
                            label=r'$w^{'+f'{l+1}'+r'}$', color=f'C{l}')
            axs[2].fill_between(np.arange(num_steps),
                                    weights_mean[l,:]+weights_std[l,:],
                                    weights_mean[l,:]-weights_std[l,:],
                                    color=f'C{l}', alpha=0.2)
        axs[2].set_title('Action Selection')
        axs[2].set_xlabel('Step')
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel(r'$w$')
        axs[2].legend(loc='lower left')
        label_subplot(fig, axs[2], 2)
        # Plot actions as a twin axis (if provided)
        if actions is not None:
            # Shape (num_models, num_steps)
            actions_mean = einops.reduce(actions, '1 s t -> t', np.mean)
            actions_std = einops.reduce(actions, '1 s t -> t', np.std)
            ax_actions = axs[2].twinx()
            # Actions are 0 or 1 for actions 1, 2, so P(A1) is 1 - actions_mean
            # Use zorder=-1 to put it in the back
            ax_actions.plot(1-actions_mean, color='C2')
            ax_actions.fill_between(np.arange(num_steps),
                                    (1-actions_mean)+actions_std,
                                    (1-actions_mean)-actions_std,
                                    color='C2', alpha=0.2)
            ax_actions.set_ylim(0, 1)
            ax_actions.set_ylabel(r'Proportion correct', color='C2')
            ax_actions.tick_params(axis='y', labelcolor='C2')
            # Put this Axes below other one (by default it goes above regardless of zorder)
            # https://stackoverflow.com/questions/38687887/how-to-define-zorder-when-using-2-y-axis
            axs[2].set_zorder(ax_actions.get_zorder()+1)
            axs[2].patch.set_visible(False)
        
        # Setting the aspect ratio doesn't work for subplots sharing axes
        square_subplots(fig) # Note, only reward prediction plot needs to be square but whatever
        plt.tight_layout()
        add_density_colorbar(fig, axs[1], weights_reward_prediction.shape[-1])

        register_fig(plotter, fig, f'Single Model All Settings {model}{" No Actions" if actions is None else ""}')


import scipy.special as scs
def get_predicted_instantaneous_drift(param_to_vary, param_vals, model, sim_kwargs):
    assert sim_kwargs['N'] == 1
    predicted_dw = []
    for v in param_vals:
        cur_kwargs = sim_kwargs.copy()
        cur_kwargs[param_to_vary] = v
        w = cur_kwargs['w_init']
        r = cur_kwargs['r']
        Rstar = cur_kwargs['Rstar']
        T_win = cur_kwargs['T_win']
        alpha = cur_kwargs['alpha']
        tau = cur_kwargs['tau']
        tau_dop = cur_kwargs['tau_dop']
        tau_eli = cur_kwargs['tau_eli']
        lambda_ = cur_kwargs['lambda_']
        if model == 'additive':
            predicted_dw.append((Rstar - w*r)*tau_dop*tau_eli*lambda_*(tau*w*(r**2)*(1 - alpha) + w*r))
        elif model == 'multiplicative':
            predicted_dw.append((Rstar - w*r)*tau_dop*tau_eli*lambda_
                                *(tau*w*(r**2)*(1 - (1 + alpha)*w) + (1 - w)*w*r))
        elif model == 'corticostriatal':
            Dplus = (Rstar*scs.gammaincc(np.floor(Rstar*T_win)+1, r*w*T_win)
                    - r*w*scs.gammaincc(np.floor(Rstar*T_win), r*w*T_win))
            Dminus = Rstar - r*w - Dplus
            predicted_dw.append(
                cur_kwargs['tau_dop'] * cur_kwargs['tau_eli'] * cur_kwargs['lambda_']
                * (Dplus*(tau*r*w*(1 - (1+alpha)*w)*r + (1-w)*w*r)
                   - Dminus*(tau*r*w*(1 - (1+alpha)*w)*r - alpha*w*w*r)))
    return np.array(predicted_dw)


def plot_reward_prediction_instantaneous_drift(weights, param_to_vary, param_vals, delay_vals, models, 
                                               sim_kwargs=None, log_x=False, scale_y=False, 
                                               save_folder=None, silent=False):
    """
    Plot instantaneous drift rate for reward prediction task
    weights should have shape (models, delay_vals, param_vals, samples, channels, N, num_steps)
    channels = 1, N = 1, num_steps = 2,
    """

    dw = weights[...,1] - weights[...,0] # Shape (models, delay_vals, param_vals, samples, 1, 1)

    if scale_y:
        dw = dw/np.array(param_vals).reshape(1,1,-1,1,1,1)

    dw_mean = einops.reduce(dw, 'm d p s 1 1 -> m d p', np.mean) # Shape (models, delay_vals, param_vals)
    dw_std = einops.reduce(dw, 'm d p s 1 1 -> m d p', np.std) # Shape (models, delay_vals, param_vals)

    num_models = len(models)

    if sim_kwargs is not None:
        predicted_dw = [get_predicted_instantaneous_drift(param_to_vary, param_vals, models[k], sim_kwargs)
                        for k in range(num_models)]
        predicted_dw = np.array(predicted_dw) # Shape (models, param_vals)

    with Plotter(save_folder=save_folder, silent=silent) as plotter:
        if param_to_vary == 'w_init':
            fig, axs = plt.subplots(ncols=num_models, squeeze=True,
                                    figsize=(1+2.5*num_models, 3))
        else:
            fig, axs = plt.subplots(ncols=num_models, sharey='row', squeeze=True,
                                    figsize=(1+2*num_models, 3))

        for k in range(num_models):
            for l in range(len(delay_vals)):
                axs[k].errorbar(param_vals, dw_mean[k,l], yerr=dw_std[k,l],
                               label=r'$T_{del}='+f'{delay_vals[l]}$ s', capsize=5)
            axs[k].axhline(0, color='black') # Plot zero line
            if sim_kwargs is not None:
                axs[k].plot(param_vals, predicted_dw[k], color='grey', linestyle='--',
                            label='Predicted', zorder=4)

            axs[k].set_title(models[k].title())
            label_subplot(fig, axs[k], k)
            if param_to_vary in param_label_map:
                axs[k].set_xlabel(param_label_map[param_to_vary])
            else:
                axs[k].set_xlabel(param_to_vary)
            if log_x:
                axs[k].set_xscale('log')
            else:
                axs[k].set_xlim(0, axs[k].get_xlim()[1])
        if scale_y:
            if param_to_vary in param_label_map:
                axs[0].set_ylabel(r'$\Delta w / ' + param_label_map[param_to_vary].split('$')[1] + '$')
            else:
                axs[0].set_ylabel(r'$\Delta w / ' + param_to_vary + '$')
        else:
            axs[0].set_ylabel(r'$\Delta w$')
        if param_to_vary == 'w_init':
            axs[-1].legend(loc='lower left')
        else:
            axs[-1].legend(loc='upper right')
        if param_to_vary == 'T_win' and param_vals[0] < 0.2:
            # Set ylims to ignore the first entry
            axs[0].set_ylim(
                1.2*min(np.min(dw_mean[:,:,1:] - dw_std[:,:,1:]), np.min(predicted_dw[:,1:])),
                1.2*max(np.max(dw_mean[:,:,1:] + dw_std[:,:,1:]), np.max(predicted_dw[:,1:]))
            )

        plt.tight_layout()

        register_fig(plotter, fig, f'Reward Prediction Instantaneous Drift')
