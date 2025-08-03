import numpy as np
import scipy.special as scs
import scipy.optimize as sco
import scipy.integrate as scint

from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

import plot_utils
from phase_plane_utils import model_funs, get_boundary_fixed_points

def dynamics_sign(r, Rstar1, Rstar2, alpha, tau, model):
    """
    Compute signs of the dynamics on the solution equilibria for additive, multiplicative, and symmetric models
    Returns 2x2 array with first axis selecting the equilibrium and second axis the eigenvalue (or 2nd order dynamics)
    """
    fp, fm = model_funs[model]
    return np.sign([
            [-(tau*r*(fp(Rstar2/r, alpha) - fm(Rstar2/r, alpha)) + fp(Rstar2/r, alpha)), Rstar1-Rstar2], # w = R*2/r, p = 0
            [-(tau*r*(fp(Rstar1/r, alpha) - fm(Rstar1/r, alpha)) + fp(Rstar1/r, alpha)), Rstar2-Rstar1] # w = R*1/r, p = 1
        ])

def vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, consts_w, consts_p, model):
    """Compute vector field for additive, multiplicative, and symmetric models"""
    dot_prod = w*r
    fp, fm = model_funs[model]

    dw = consts_w*(p*Rstar1 + (1-p)*Rstar2 - dot_prod)*(tau*(fp(w, alpha) - fm(w, alpha))*r*dot_prod + r*fp(w, alpha)*w)
    dp = consts_p*p*(1-p)*(p*(Rstar1 - dot_prod) - (1-p)*(Rstar2 - dot_prod))

    return dw, dp

def corticostriatal_vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, T_win, consts_w, consts_p):
    """Compute vector field for corticostriatal model"""
    dot_prod = w*r 
    Dplus1 = (Rstar1*scs.gammaincc(np.floor(Rstar1*T_win)+1, dot_prod*T_win)
            - dot_prod*scs.gammaincc(np.floor(Rstar1*T_win), dot_prod*T_win))
    Dplus2 = (Rstar2*scs.gammaincc(np.floor(Rstar2*T_win)+1, dot_prod*T_win)
            - dot_prod*scs.gammaincc(np.floor(Rstar2*T_win), dot_prod*T_win))
    Dplus = p*Dplus1 + (1-p)*Dplus2
    Dminus = p*Rstar1 + (1-p)*Rstar2 - dot_prod - Dplus
    dw = consts_w*(Dplus*(tau*dot_prod*(1 - (1+alpha)*w)*r + (1-w)*w*r)
        - Dminus*(tau*dot_prod*(1 - (1+alpha)*w)*r - alpha*w*w*r))
    dp = consts_p*p*(1-p)*(p*(Rstar1 - dot_prod) - (1-p)*(Rstar2 - dot_prod))

    return dw, dp


def plot_phase_plane(ax, model, model_kwargs, points=10, color=None, marker='P'):
    """Plot a phase portrait on a given Axes object"""
    Rstar1, Rstar2 = model_kwargs['Rstar'][0], model_kwargs['Rstar'][1]
    r = model_kwargs['r']
    alpha = model_kwargs['alpha']
    tau = model_kwargs['tau']
    T_win = model_kwargs['T_win']
    consts_w = model_kwargs['lambda_']*model_kwargs['r_dop']*model_kwargs['tau_eli']*model_kwargs['tau_dop']/model_kwargs['N']
    consts_p = model_kwargs['lambda_bar']*model_kwargs['beta']*model_kwargs['r_dop']*model_kwargs['tau_dop']

    w, p = np.meshgrid(np.linspace(0,1,points), np.linspace(0,1,points))
    w, p = w.T, p.T

    if model in model_funs:
        dw, dp = vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, consts_w, consts_p, model)
        fp_dynamics = dynamics_sign(r, Rstar1, Rstar2, alpha, tau, model)
    elif model == 'corticostriatal':
        dw, dp = corticostriatal_vector_field(w, p, r, Rstar1, Rstar2, alpha, tau, T_win, consts_w, consts_p)
        fp_dynamics = None
    else:
        dw = None
        dp = None
        fp_dynamics = None

    # Plot the quiver field
    if dw is not None and dp is not None:
        if color is None:
            lengths = np.sqrt(dw**2 + dp**2)
            ax.quiver(w, p, dw, dp, lengths, zorder=0)
        else:
            ax.quiver(w, p, dw, dp, color=color, zorder=0)

    all_fps = []

    # Plot solution equilibria
    if fp_dynamics is not None:
        solution_fps = [[Rstar2/r, 0], [Rstar1/r, 1]]
        all_fps.extend(solution_fps)
        for fp, sgn in zip(solution_fps, fp_dynamics):
            if np.all(sgn < 0):
                fp_color = plot_utils.colors['stable_fp']
                fp_label = 'Fixed point (stable)'
            elif np.all(sgn > 0):
                fp_color = plot_utils.colors['unstable_fp']
                fp_label = 'Fixed point (unstable)'
            else:
                fp_color = plot_utils.colors['other_fp']
                fp_label = 'Fixed point (other)'
            ax.scatter(fp[0], fp[1], marker=marker, color=fp_color, label=fp_label,
                       zorder=10).set_clip_on(False)

    # Plot the equilibria along w=0
    if model in ['additive', 'symmetric', 'multiplicative', 'corticostriatal']:
        w0_fps = [[0, 0], [0, 1], [0, Rstar2/(Rstar1+Rstar2)]]
        all_fps.extend(w0_fps)
        if model == 'additive' or model == 'symmetric':
            stability_condition = tau*(alpha-1) < 1/r
            if stability_condition:
                fp_colors = [plot_utils.colors['other_fp'], plot_utils.colors['other_fp'], plot_utils.colors['unstable_fp']]
                fp_labels = ['Fixed point (other)', 'Fixed point (other)', 'Fixed point (unstable)']
            else:
                fp_colors = [plot_utils.colors['stable_fp'], plot_utils.colors['stable_fp'], plot_utils.colors['other_fp']]
                fp_labels = ['Fixed point (stable)', 'Fixed point (stable)', 'Fixed point (other)']
        elif model == 'multiplicative' or model == 'corticostriatal':
            fp_colors = [plot_utils.colors['other_fp'], plot_utils.colors['other_fp'], plot_utils.colors['unstable_fp']]
            fp_labels = ['Fixed point (other)', 'Fixed point (other)', 'Fixed point (unstable)']
        for fp, fp_color, fp_label in zip(w0_fps, fp_colors, fp_labels):
            ax.scatter(fp[0], fp[1], marker=marker, color=fp_color, label=fp_label,
                        zorder=10).set_clip_on(False)
    # Plot the equilibria along w=1
    if model == 'symmetric':
        w1_fps = [[1, 0], [1, 1], [1, (Rstar2 - r)/(Rstar1 + Rstar2 - 2*r)]]
        all_fps.extend(w1_fps)
        stability_condition = tau*(alpha-1) < 1/r
        if stability_condition:
            fp_colors = [plot_utils.colors['unstable_fp'], plot_utils.colors['unstable_fp'], plot_utils.colors['other_fp']]
            fp_labels = ['Fixed point (unstable)', 'Fixed point (unstable)', 'Fixed point (other)']
        else:
            fp_colors = [plot_utils.colors['other_fp'], plot_utils.colors['other_fp'], plot_utils.colors['stable_fp']]
            fp_labels = ['Fixed point (other)', 'Fixed point (other)', 'Fixed point (stable)']
        for fp, fp_color, fp_label in zip(w1_fps, fp_colors, fp_labels):
            ax.scatter(fp[0], fp[1], marker=marker, color=fp_color, label=fp_label,
                       zorder=10).set_clip_on(False)

    # Plot the multiplicative model's extra equilibria
    if model == 'multiplicative':
        w0 = (tau*r + 1)/(tau*(1 + alpha)*r + 1)
        for cur_p, cur_Rstar in [[0, Rstar2], [1, Rstar1]]:
            all_fps.append([w0, cur_p])
            if cur_Rstar < r*w0:
                fp_color = plot_utils.colors['unstable_fp']
                fp_label = 'Fixed point (unstable)'
            elif cur_Rstar > r*w0:
                fp_color = plot_utils.colors['stable_fp']
                fp_label = 'Fixed point (stable)'
            else:
                fp_colors.append(plot_utils.colors['other_fp'])
                fp_labels.append('Fixed point (other)')
            ax.scatter(w0, cur_p, marker=marker, color=fp_color, label=fp_label,
                       zorder=10).set_clip_on(False)


    # Now plot any other fixed points along the boundary
    if model in model_funs:
        dw_fun = lambda x: np.array(vector_field(x[0], x[1], r, Rstar1, Rstar2, alpha, tau, consts_w, consts_p, model))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun, fps_to_filter=all_fps)
    elif model == 'corticostriatal':
        dw_fun = lambda x: np.array(corticostriatal_vector_field(x[0], x[1], r, Rstar1, Rstar2, alpha, tau, T_win, consts_w, consts_p))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun)
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
        dynamics_fun = lambda t,y: vector_field(y[0], y[1], r, Rstar1, Rstar2, alpha, tau, consts_w, consts_p, model)
    elif model == 'corticostriatal':
        dynamics_fun = lambda t,y: corticostriatal_vector_field(y[0], y[1], r, Rstar1, Rstar2, alpha, tau, T_win, consts_w, consts_p)
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

def plot_center_manifold(ax, model, model_kwargs, manifold_points=25, 
                         color=plot_utils.colors['center_manifold'], manifold_length=0.2):
    """Plot the center manifold on an existing Axes object"""
    Rstar1, Rstar2 = model_kwargs['Rstar'][0], model_kwargs['Rstar'][1]
    r = model_kwargs['r']
    alpha = model_kwargs['alpha']
    tau = model_kwargs['tau']
    r_dop = model_kwargs['r_dop']
    tau_dop = model_kwargs['tau_dop']
    tau_eli = model_kwargs['tau_eli']
    lambda_ = model_kwargs['lambda_']
    lambda_bar = model_kwargs['lambda_bar']
    beta = model_kwargs['beta']

    if model in model_funs:
        fp, fm = model_funs[model]
        for w_star, p_star in [[Rstar2/r, 0], [Rstar1/r, 1]]:
            # Compute eigenvector and eigenvalue
            v1 = tau*r*r*w_star*(fp(w_star, alpha) - fm(w_star, alpha)) + fp(w_star, alpha)*w_star*r
            if v1 == 0:
                print(f'v1 = 0, equilibrium at p={p_star} is degenerate')
                return
            Lambda = -r_dop*tau_dop*tau_eli*lambda_*v1*r
            # Compute center manifold coefficient
            h = 2*lambda_bar*beta*r_dop*tau_dop/(Lambda*v1*r)
            # Compute the points on the center manifold
            if p_star == 0:
                # Parameterize the manifold so that the length of the curve is fixed
                max_val = sco.root_scalar(lambda x: (x/r + h*v1*(x**2))**2 + (x/(Rstar1 - Rstar2))**2 - manifold_length**2,
                                                bracket=(0, np.abs(Rstar1 - Rstar2)*manifold_length)).root
                manifold_vals = np.linspace(0, max_val, manifold_points)
                w_manifold = w_star + manifold_vals/r + h*v1*(manifold_vals**2)
                p_manifold = p_star + manifold_vals/(Rstar1 - Rstar2)
                flow_direction = Rstar1 > Rstar2
            else:
                # Parameterize the manifold so that the length of the curve is fixed
                max_val = sco.root_scalar(lambda x: ((-x/r + h*v1*(x**2))**2 + (x/(Rstar1 - Rstar2))**2 - manifold_length**2),
                                                bracket=(0, np.abs(Rstar1 - Rstar2)*manifold_length)).root
                manifold_vals = np.linspace(0, max_val, manifold_points)
                w_manifold = w_star - manifold_vals/r + h*v1*(manifold_vals**2)
                p_manifold = p_star - manifold_vals/(Rstar1 - Rstar2)
                flow_direction = Rstar1 < Rstar2
            # Plot the manifold curve
            lines = ax.plot(w_manifold, p_manifold, color=color, zorder=9)
            for line in lines:
                line.set_clip_on(False)
            # Add arrow at end of curve
            angle = np.arctan2(p_manifold[-1] - p_manifold[-2], w_manifold[-1] - w_manifold[-2]) # For some reason it's (y,x)
            transform = Affine2D().rotate(angle)
            lines = ax.plot(w_manifold[-1], p_manifold[-1], color=color, zorder=9,
                    marker=MarkerStyle('>' if flow_direction else '<', 'full', transform))
            for line in lines:
                line.set_clip_on(False)
    else:
        print(f"Center manifold not implemented for model {model}")
        return