import numpy as np
import scipy.integrate as scint

from poisson_model import get_expected_action_prob

import plot_utils
from phase_plane_utils import model_funs, get_boundary_fixed_points



def vector_field(w1, w2, r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, model, a_sel):
    """Compute vector field for additive, multiplicative, and symmetric models"""
    dot_prod1 = w1*r
    dot_prod2 = w2*r
    fp, fm = model_funs[model]

    if isinstance(dot_prod1, np.ndarray):
        exp_P1 = np.zeros_like(dot_prod1)
        for coords in np.ndindex(exp_P1.shape):
            exp_P1[coords] = get_expected_action_prob(dot_prod1[coords], dot_prod2[coords], beta, T_win)
    else:
        exp_P1 = get_expected_action_prob(dot_prod1, dot_prod2, beta, T_win)

    dw1 = consts*exp_P1*(1 - exp_P1)*(Rstar1 - Rstar2)*(tau*(fp(w1, alpha) - fm(w1, alpha))*r*dot_prod1*(a_sel**2) + r*fp(w1, alpha)*w1*a_sel)
    dw2 = consts*exp_P1*(1 - exp_P1)*(Rstar2 - Rstar1)*(tau*(fp(w2, alpha) - fm(w2, alpha))*r*dot_prod2*(a_sel**2) + r*fp(w2, alpha)*w2*a_sel)

    return dw1, dw2

def corticostriatal_vector_field(w1, w2, r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, a_sel):
    """Compute vector field for corticostriatal model"""
    dot_prod1 = w1*r
    dot_prod2 = w2*r

    if isinstance(dot_prod1, np.ndarray):
        exp_P1 = np.zeros_like(dot_prod1)
        for coords in np.ndindex(exp_P1.shape):
            exp_P1[coords] = get_expected_action_prob(dot_prod1[coords], dot_prod2[coords], beta, T_win)
    else:
        exp_P1 = get_expected_action_prob(dot_prod1, dot_prod2, beta, T_win)

    if Rstar1 >= Rstar2:
        dw1 = consts*exp_P1*(1 - exp_P1)*(Rstar1 - Rstar2)*(tau*dot_prod1*(1 - (1+alpha)*w1)*r*(a_sel**2) + (1-w1)*w1*r*a_sel)
        dw2 = consts*exp_P1*(1 - exp_P1)*(Rstar1 - Rstar2)*(tau*dot_prod2*(1 - (1+alpha)*w2)*r*(a_sel**2) - alpha*w2*w2*r*a_sel)
    else:
        dw1 = consts*exp_P1*(1 - exp_P1)*(Rstar2 - Rstar1)*(tau*dot_prod1*(1 - (1+alpha)*w1)*r*(a_sel**2) - alpha*w1*w1*r*a_sel)
        dw2 = consts*exp_P1*(1 - exp_P1)*(Rstar2 - Rstar1)*(tau*dot_prod2*(1 - (1+alpha)*w2)*r*(a_sel**2) + (1-w2)*w2*r*a_sel)

    return dw1, dw2

def plot_phase_plane(ax, model, model_kwargs, points=10, color=None, marker='P'):
    """Plot a phase portrait on a given Axes object"""
    Rstar1, Rstar2 = model_kwargs['Rstar'][0], model_kwargs['Rstar'][1]
    r = model_kwargs['r']
    alpha = model_kwargs['alpha']
    tau = model_kwargs['tau']
    T_win = model_kwargs['T_win']
    consts = model_kwargs['lambda_']*model_kwargs['r_dop']*model_kwargs['tau_eli']*model_kwargs['tau_dop']/model_kwargs['N']
    a_sel = model_kwargs['a_sel']
    beta = model_kwargs['beta']

    w1, w2 = np.meshgrid(np.linspace(0,1,points), np.linspace(0,1,points))
    w1, w2 = w1.T, w2.T

    if model in model_funs:
        dw1, dw2 = vector_field(w1, w2, r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, model, a_sel)
    elif model == 'corticostriatal':
        dw1, dw2 = corticostriatal_vector_field(w1, w2, r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, a_sel)
    else:
        dw1 = None
        dw2 = None

    # Plot the quiver field
    if dw1 is not None and dw2 is not None:
        if color is None:
            lengths = np.sqrt(dw1**2 + dw2**2)
            ax.quiver(w1, w2, dw1, dw2, lengths, zorder=0)
        else:
            ax.quiver(w1, w2, dw1, dw2, color=color, zorder=0)

    all_fps = []
    
    # Plot equilibrium at the origin
    if model in ['additive', 'symmetric']:
        all_fps.append([0,0])
        fp_color = plot_utils.colors['other_fp']
        fp_label = 'Fixed point (other)'
        ax.scatter(0, 0, marker=marker, color=fp_color, label=fp_label, zorder=10).set_clip_on(False)
    elif model == 'corticostriatal':
        all_fps.append([0,0])
        fp_color = plot_utils.colors['unstable_fp']
        fp_label = 'Fixed point (unstable)'
        ax.scatter(0, 0, marker=marker, color=fp_color, label=fp_label, zorder=10).set_clip_on(False)
    
    # Plot equilibria at w=1
    if model == 'symmetric':
        w1_fps = [[0,1], [1,0], [1,1]]
        all_fps.extend(w1_fps)
        if Rstar1 >= Rstar2:
            stability_condition = a_sel*tau*(alpha-1) < 1/r
        else:
            stability_condition = a_sel*tau*(alpha-1) > 1/r
        if stability_condition:
            fp_colors = [plot_utils.colors['unstable_fp'], plot_utils.colors['stable_fp'], plot_utils.colors['other_fp']]
            fp_labels = ['Fixed point (unstable)', 'Fixed point (stable)', 'Fixed point (other)']
        else:
            fp_colors = [plot_utils.colors['stable_fp'], plot_utils.colors['unstable_fp'], plot_utils.colors['other_fp']]
            fp_labels = ['Fixed point (stable)', 'Fixed point (unstable)', 'Fixed point (other)']
        for fp, fp_color, fp_label in zip(w1_fps, fp_colors, fp_labels):
            ax.scatter(fp[0], fp[1], marker=marker, color=fp_color, label=fp_label, zorder=10).set_clip_on(False)
        
    # Plot the corticostriatal model's extra equilibria
    if model == 'corticostriatal':
        w1_star = (a_sel*tau*r + 1)/(a_sel*tau*(1+alpha)*r + 1)
        w2_star = a_sel*tau*r/(a_sel*tau*(1+alpha)*r + alpha)
        if Rstar1 >= Rstar2:
            cs_fps = [[w1_star, w2_star], [w1_star, 0], [0, w2_star]]
        else:
            # Reverse w1, w2
            cs_fps = [[w2_star, w1_star], [w2_star, 0], [0, w1_star]]
        all_fps.extend(cs_fps)
        fp_colors = [plot_utils.colors['stable_fp'], plot_utils.colors['other_fp'], plot_utils.colors['other_fp']]
        fp_labels = ['Fixed point (stable)', 'Fixed point (other)', 'Fixed point (other)']
        for fp, fp_color, fp_label in zip(cs_fps, fp_colors, fp_labels):
            ax.scatter(fp[0], fp[1], marker=marker, color=fp_color, label=fp_label, zorder=10).set_clip_on(False)

    # Now plot any other fixed points along the boundary
    if model in model_funs:
        # Don't include scaling factors so that it's not affected by learning rates, etc.
        dw_fun = lambda x: np.array(vector_field(x[0], x[1], r, Rstar1, Rstar2, alpha, tau, 1, beta, T_win, model, a_sel))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun, fps_to_filter=all_fps)
    elif model == 'corticostriatal':
        # Don't include scaling factors so that it's not affected by learning rates, etc.
        dw_fun = lambda x: np.array(corticostriatal_vector_field(x[0], x[1], r, Rstar1, Rstar2, alpha, tau, 1, beta, T_win, a_sel))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun, fps_to_filter=all_fps)
    else:
        stable_fps = []
        unstable_fps = []
        other_fps = [] 
    
    # Plot with facecolors='none' to make unfilled markers
    if len(stable_fps) > 0:
        ax.scatter(np.array(stable_fps)[:,0], np.array(stable_fps)[:,1], 
                   marker=marker, color=plot_utils.colors['stable_fp'], label='Fixed point (stable)', zorder=10, facecolors='none').set_clip_on(False)
    if len(unstable_fps) > 0:
        ax.scatter(np.array(unstable_fps)[:,0], np.array(unstable_fps)[:,1], 
                   marker=marker, color=plot_utils.colors['unstable_fp'], label='Fixed point (unstable)', zorder=10, facecolors='none').set_clip_on(False)
    if len(other_fps) > 0:
        ax.scatter(np.array(other_fps)[:,0], np.array(other_fps)[:,1], 
                   marker=marker, color=plot_utils.colors['other_fp'], label='Fixed point (other)', zorder=10, facecolors='none').set_clip_on(False)

    # Plot IVP solution
    if model in model_funs:
        dynamics_fun = lambda t,y: vector_field(y[0], y[1], r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, model, a_sel)
    elif model == 'corticostriatal':
        dynamics_fun = lambda t,y: corticostriatal_vector_field(y[0], y[1], r, Rstar1, Rstar2, alpha, tau, consts, beta, T_win, a_sel)
    else:
        dynamics_fun = None
    
    if dynamics_fun is not None:
        max_t = model_kwargs['num_steps']/model_kwargs['r_dop']
        sol = scint.solve_ivp(dynamics_fun, [0, max_t], [0.5, 0.5], t_eval=np.linspace(0, max_t, 1000))
        if not sol.success:
            print(f'IVP solution failed for model {model} with status {sol.status} and message {sol.message}')
        else:
            ax.plot(sol.y[0], sol.y[1], color=plot_utils.colors['solution'], zorder=9, alpha=0.75)

    # Let the calling function deal with axis labels, bounds, etc. because it might want to
    # plot other things (like sample paths) on top