
import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import einops

plt.rcParams.update({
    'text.usetex': True,
    'savefig.dpi': 600,
})


subplot_labels = 'abcdefghijklmnopqrstuvwxyz'

colorblind_palette = sns.color_palette("colorblind", as_cmap=True)
colors = {
    'weights': [colorblind_palette[0], colorblind_palette[3]], # Blue, Orange
    'actions': colorblind_palette[2], # Green
    'p': colorblind_palette[1], # Yellow
    'w_init_vals': [colorblind_palette[0], colorblind_palette[3], colorblind_palette[2]], # Blue, Orange, Green
    'param_vals': [colorblind_palette[0], colorblind_palette[3], colorblind_palette[2]], # Blue, Orange, Green
    'predicted': colorblind_palette[7], # Gray
    'stable_fp': 'black',
    'unstable_fp': colorblind_palette[3], # Orange
    'other_fp': colorblind_palette[2], # Green
    'neutral_marker': colorblind_palette[7], # Gray
    'w_init_marker': 'black',
    'center_manifold': colorblind_palette[4], # Purple
    'solution': 'black',
    'vector_field': 'dimgray',
}


density_cmap = plt.get_cmap('viridis')
# Truncate the colormap to avoid lightness extremes
density_cmap = plt.cm.colors.ListedColormap(density_cmap(np.linspace(0.15, 0.85, 256)))


param_label_map = {
    'r': r'$r$',
    'tau_eli': r'$\tau_{eli}$',
    'tau_dop': r'$\tau_{dop}$',
    'tau': r'$\tau$',
    'alpha': r'$\alpha$',
    'gamma': r'$\gamma$',
    'T_win': r'$T_{win}$',
    'T_del': r'$T_{del}$',
    'w_init': r'$w_{init}$',
    'Rstar': r'$R^*$',
    'Rstar1': r'$R^*_1$',
    'Rstar2': r'$R^*_2$',
    'eps': r'$\epsilon$',
    'beta': r'$\beta$',
}
param_unit_map = {
    'r': '',
    'tau_eli': ' s',
    'tau_dop': ' s',
    'tau': ' s',
    'alpha': '',
    'gamma': '',
    'T_win': ' s',
    'T_del': ' s',
    'w_init': '',
    'Rstar': '',
    'Rstar1': '',
    'Rstar2': '',
    'eps': ' s',
    'beta': '',
}

fig_save_formats = ['png', 'eps']

def square_subplots(fig):
    """Manually ensure subplots are square to get around various issues with setting the aspect ratio"""
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
    """Add a label to a subplot"""
    trans = mtransforms.ScaledTranslation(scale*0.175, -scale*0.105, #scale*12.5/72, -scale*7.5/72,
                                          fig.dpi_scale_trans)
    ax.text(0.0, scale, subplot_labels[label_index], 
        transform=ax.transAxes + trans,
        fontsize='medium', verticalalignment='top',
        horizontalalignment='center',
        # Params to match the default legend params
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'edgecolor': '0.8', 
              'alpha': 0.5},
        ) 

def plot_density(ax, weights, bins):
    """
    Adds density plot to an Axes object
    weights should have shape (num_samples, N or channels, num_steps)
    or shape (num_param_vals, num_samples, N or channels, num_steps)
    N or channels should be 2
    """
    num_steps = weights.shape[-1]
    if len(weights.shape) == 4:
        num_samples = weights.shape[1]
        weights = einops.rearrange(weights, 'p s N t -> (p s) N t')
    else:
        num_samples = weights.shape[0]
    bin_arr = np.linspace(0,1,bins+1)

    # Keep running sums
    alpha_sum = np.zeros((bins, bins,1))
    weighted_img_sum = np.zeros((bins, bins, 3))
    for k in range(num_steps):
        # Use alpha to indicate density, and color to indicate time
        cur_hist = np.histogram2d(weights[:,0,k], weights[:,1,k], bins=bin_arr)[0].T
        # This has shape (x, y, 4) giving RGBA (with values in [0,1])
        cur_img = density_cmap(k*np.ones_like(cur_hist)/num_steps, alpha=cur_hist/num_samples)
        # Average the time weighted by samples (alpha)
        alpha_sum += cur_img[:,:,3:4] # Shape (x, y, 1)
        weighted_img_sum += cur_img[:,:,:3]*cur_img[:,:,3:4] # Shape (x, y, 3)
    img = np.divide(weighted_img_sum, alpha_sum, out=np.zeros((bins,bins,3)), where=alpha_sum!=0)
    # Normalize
    scale = num_samples/100 # This has units of steps
    alpha = np.clip(alpha_sum/scale, 0, 1) # Shape (x, y, 1)
    ax.imshow(np.full((bins, bins, 3), 255, dtype=np.uint8), 
                extent=(0,1,0,1), zorder=-2) # Add white background
    # For some reason using alpha parameter of imshow doesn't work so do this instead
    img = np.concatenate([img, alpha], axis=-1)
    ax.imshow(img, aspect='equal', extent=(0,1,0,1), origin='lower', zorder=-1)

    # Mark the initial point of all the weights
    ax.scatter(weights[0,0,0], weights[0,1,0], marker='x', color=colors['w_init_marker'], linewidth=1)
    # Let the calling function deal with axis labels, bounds, etc.

def add_density_colorbar(fig, axs, max_step, add_legend=False, 
                         legend_points=['stable', 'unstable', 'other', 'winit']):
    """Add a colorbar to a density/phase plane plot"""
    # Extend the figsize for colorbar
    fig.set_size_inches(fig.get_size_inches()[0]+1.5, fig.get_size_inches()[1])
    if not add_legend:
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,max_step), cmap=density_cmap), 
                     ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs, 
                     shrink=0.9, label='Step', pad=0.025, fraction=0.1)
    else:
        legend_elements = []
        if 'stable' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor=colors['stable_fp'], marker='P', 
                                          markersize=12, label='Stable'))
        if 'unstable' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor=colors['unstable_fp'], marker='P', 
                                          markersize=12, label='Unstable'))
        if 'other' in legend_points:
            legend_elements.append(Line2D([0], [0], color='white', markerfacecolor=colors['other_fp'], marker='P', 
                                          markersize=12, label='Other'))
        if 'numerical/analytical' in legend_points:
            legend_elements.append(Line2D([0],[0], color='white', markerfacecolor='none', markeredgecolor=colors['neutral_marker'], marker='P', 
                                          markersize=10, label='Numerical'))
            legend_elements.append(Line2D([0],[0], color='white', markerfacecolor=colors['neutral_marker'], marker='P', 
                                          markersize=12, label='Analytical'))
        if 'winit' in legend_points:
            legend_elements.append(Line2D([0], [0], color=colors['w_init_marker'], marker='x', 
                                          linewidth=0, markersize=10, label=r'$w_{init}$'))
        if 'center manifold' in legend_points:
            legend_elements.append(Line2D([0,1], [0,0], color=colors['center_manifold'], linewidth=2, label='Center\nmanifold'))
        if 'solution' in legend_points:
            legend_elements.append(Line2D([0,1], [0,0], color=colors['solution'], alpha=0.75, linewidth=2, label='Solution'))
        upper_right_bbox = axs[0,-1].get_position() if isinstance(axs, np.ndarray) else axs.get_position()
        ur_y = upper_right_bbox.y1
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, ur_y))
        
        fig_h = fig.get_size_inches()[1]
        ur_h_in = (1 - ur_y)*fig_h
        num_legend_lines = len(legend_elements)
        if 'center manifold' in legend_points:
            num_legend_lines += 1

        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,max_step), cmap=density_cmap), 
                     ax=axs.ravel().tolist() if isinstance(axs, np.ndarray) else axs, 
                     shrink=(fig_h - ur_h_in - 0.2*num_legend_lines)/fig_h,
                     label='Step', pad=0.025, fraction=0.1, anchor=(0,0),
                     aspect=20 if isinstance(axs, np.ndarray) and axs.shape[0] == 3 else 10)


class Plotter:
    """
    Context manager to handle saving or displaying figures
    """
    def __init__(self, save_folder=None, silent=False):
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.silent = silent

        if silent:
            plt.switch_backend('Agg')
        else:
            plt.switch_backend('TkAgg')

    def __enter__(self):
        return self

    def register(self, fig, fname, formats=None, **kwargs):
        if self.save_folder is not None:
            if formats is None:
                fig.savefig(os.path.join(self.save_folder, fname), **kwargs)
            else:
                for fmt in formats:
                    if fmt == 'eps':
                        # For eps files, first generate a pdf then convert to eps
                        fig.savefig(os.path.join(self.save_folder, fname + '.pdf'), **kwargs)
                        subprocess.call(['pdf2ps', 
                                         os.path.join(self.save_folder, fname + '.pdf'),
                                         os.path.join(self.save_folder, fname + '.eps')])
                    else:
                        fig.savefig(os.path.join(self.save_folder, fname + '.' + fmt), **kwargs)

        if self.silent:
            plt.close(fig)


    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None: # An exception was raised
            plt.close('all')
        else:
            if not self.silent:
                plt.show()
            else:
                plt.close('all')
        # Note: returns None, which is falsy
        # This makes the 'with' statement reraise any exception as it's not handled here