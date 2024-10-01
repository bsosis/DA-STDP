import numpy as np
import scipy.special as scs
import scipy.optimize as sco

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

model_funs = { # (f+, f-)
    'additive': 
        (lambda x, alpha: 1,
        lambda x, alpha: alpha),
    'multiplicative': 
        (lambda x, alpha: 1 - x,
        lambda x, alpha: alpha*x),
    'symmetric':
        (lambda x, alpha: x*(1 - x),
        lambda x, alpha: alpha*x*(1 - x)),
}

def simple_eig(r1, r2, Rstar, alpha, tau, model):
    """Compute eigenvalue of Jacobian for additive, multiplicative, and symmetric models"""
    fp, fm = model_funs[model]

    eig = lambda x, y: -(tau*Rstar*(fp(x, alpha)-fm(x, alpha))*r1**2 + (1/2)*fp(x, alpha)*x*r1**2
                        + tau*Rstar*(fp(y, alpha)-fm(y, alpha))*r2**2 + (1/2)*fp(y, alpha)*y*r2**2)
    return eig

def simple_vector_field(w1, w2, r1, r2, Rstar, alpha, tau, model):
    """Compute vector field for additive, multiplicative, and symmetric models"""
    dot_prod = w1*r1 + w2*r2
    fp, fm = model_funs[model]

    dw1 = (Rstar - dot_prod/2)*(tau*(fp(w1, alpha) - fm(w1, alpha))*r1*dot_prod + r1*fp(w1, alpha)*w1)
    dw2 = (Rstar - dot_prod/2)*(tau*(fp(w2, alpha) - fm(w2, alpha))*r2*dot_prod + r2*fp(w2, alpha)*w2)
    return dw1, dw2

def corticostriatal_vector_field(w1, w2, r1, r2, Rstar, alpha, tau, T_win):
    """Compute vector field for corticostriatal model"""
    dot_prod = w1*r1 + w2*r2 # Supports w1, w2 of any shape
    rpost = 0.5*dot_prod
    Dplus = (Rstar*scs.gammaincc(np.floor(Rstar*T_win)+1, rpost*T_win)
            - rpost*scs.gammaincc(np.floor(Rstar*T_win), rpost*T_win))
    Dminus = Rstar - rpost - Dplus
    dw1 = (Dplus*(tau*dot_prod*(1 - (1+alpha)*w1)*r1 + (1-w1)*w1*r1)
        - Dminus*(tau*dot_prod*(1 - (1+alpha)*w1)*r1 - alpha*w1*w1*r1))
    dw2 = (Dplus*(tau*dot_prod*(1 - (1+alpha)*w2)*r2 + (1-w2)*w2*r2)
        - Dminus*(tau*dot_prod*(1 - (1+alpha)*w2)*r2 - alpha*w2*w2*r2))
    return dw1, dw2

def vector_field_fast_switching(w1, w2, r, Rstar_vals, alpha, tau, model, T_win=None):
    """
    Compute vector field oin frequent task switching settings by averaging the vector fields being switched between
    """
    if model in model_funs:
        dw1_A, dw2_A = simple_vector_field(w1, w2, r[0][0], r[0][1], Rstar_vals[0], alpha, tau, model)
        dw1_B, dw2_B = simple_vector_field(w1, w2, r[1][0], r[1][1], Rstar_vals[1], alpha, tau, model)
    elif model == 'corticostriatal':
        dw1_A, dw2_A = corticostriatal_vector_field(w1, w2, r[0][0], r[0][1], Rstar_vals[0], alpha, tau, T_win)
        dw1_B, dw2_B = corticostriatal_vector_field(w1, w2, r[1][0], r[1][1], Rstar_vals[1], alpha, tau, T_win)
    return (dw1_A + dw1_B)/2, (dw2_A + dw2_B)/2

def get_boundary_fixed_points(dw_fun, Rstar=None, r=None, points=100, eps=1e-8):
    """
    Get fixed points (including artificial ones caused by clipping) along the boundary of [0,1]^2
    dw_fun should take in x=[w1, w2] and return dx=[dw1, dw2], with vectorization in the 2nd dimension
    """

    stable_fps = []
    unstable_fps = []
    other_fps = []
    
    # First examine boundaries, away from corners
    t_vals = np.arange(1/(points+1), 1-1/(points+2), 1/(points+1)).reshape(1,-1)
    for offset, tangent, inward_normal in [(np.array([0,0]), np.array([1,0]), np.array([0,1])),
                                           (np.array([0,0]), np.array([0,1]), np.array([1,0])),
                                           (np.array([1,1]), np.array([-1,0]), np.array([0,-1])),
                                           (np.array([1,1]), np.array([0,-1]), np.array([-1,0]))]:
        offset = offset.reshape(2,1)
        tangent = tangent.reshape(2,1)
        inward_normal = inward_normal.reshape(2,1)
        # We want points where dw_tangent = 0 and dw_inwards <= 0
        dw_inwards = lambda t: np.sum(dw_fun(offset + t*tangent)*inward_normal, axis=0)
        dw_tangent = lambda t: np.sum(dw_fun(offset + t*tangent)*tangent, axis=0)
        # Find changes in sign of dt_tangent
        # Note, this will miss fixed points that look like -> . -> or changes on small scales
        tangent_along_line = dw_tangent(t_vals).flatten()
        sign_changes = tangent_along_line[:-1]*tangent_along_line[1:] < 0
        # Refine points found
        candidates = []
        for i in range(len(sign_changes)):
            if sign_changes[i]:
                candidate = sco.root_scalar(dw_tangent, bracket=(t_vals[0,i], t_vals[0,i+1])).root
                # Check that it's pointing in the right direction
                if dw_inwards(candidate) <= 0:
                    candidates.append(candidate)
        # Check stability of the points found
        for t in candidates:
            pos_side = dw_tangent(t + eps)
            neg_side = dw_tangent(t - eps)
            if pos_side < 0 and neg_side > 0:
                stable_fps.append(offset.flatten() + t*tangent.flatten())
            elif pos_side > 0 and neg_side < 0:
                unstable_fps.append(offset.flatten() + t*tangent.flatten())
            else:
                other_fps.append(offset.flatten() + t*tangent.flatten())
    # Now check corners
    for corner, ax1, ax2 in [(np.array([0,0]), np.array([1,0]), np.array([0,1])),
                             (np.array([1,0]), np.array([-1,0]), np.array([0,1])),
                             (np.array([1,1]), np.array([-1,0]), np.array([0,-1])),
                             (np.array([0,1]), np.array([1,0]), np.array([0,-1]))]:
        dw = np.linalg.norm(dw_fun(corner))
        dw_ax1 = np.dot(dw_fun(corner + eps*ax1), ax1)
        dw_ax2 = np.dot(dw_fun(corner + eps*ax2), ax2)
        if dw_ax1 < 0 and dw_ax2 < 0:
            stable_fps.append(corner)
        elif dw < eps and dw_ax1 > 0 and dw_ax2 > 0:
            unstable_fps.append(corner)
        elif dw < eps:
            other_fps.append(corner)

    # Filter out points on the solution plane
    if Rstar is not None and r is not None:
        stable_fps = [fp for fp in stable_fps if not np.isclose(np.dot(fp, r)/len(r), Rstar, atol=eps)]
        unstable_fps = [fp for fp in unstable_fps if not np.isclose(np.dot(fp, r)/len(r), Rstar, atol=eps)]
        other_fps = [fp for fp in other_fps if not np.isclose(np.dot(fp, r)/len(r), Rstar, atol=eps)] 
        
    return stable_fps, unstable_fps, other_fps


marker = 'P'

def plot_phase_plane(ax, model, alpha, tau, r, Rstar, T_win=None, points=10, color=None):
    """Plot a phase portrait on a given Axes object"""

    r1, r2 = r[0], r[1]
    w1, w2 = np.meshgrid(np.linspace(0,1,points), np.linspace(0,1,points))
    w1, w2 = w1.T, w2.T

    if model in model_funs:
        dw1, dw2 = simple_vector_field(w1, w2, r1, r2, Rstar, alpha, tau, model)
        eig = simple_eig(r1, r2, Rstar, alpha, tau, model)
    elif model == 'corticostriatal':
        dw1, dw2 = corticostriatal_vector_field(w1, w2, r1, r2, Rstar, alpha, tau, T_win)
        eig = None
    else:
        dw1 = None
        dw2 = None
        eig = None

    # Plot the quiver field
    if dw1 is not None and dw2 is not None:
        if color is None:
            lengths = np.sqrt(dw1**2 + dw2**2)
            ax.quiver(w1, w2, dw1, dw2, lengths, zorder=0)
        else:
            ax.quiver(w1, w2, dw1, dw2, color=color, zorder=0)

    # Plot solution line
    w1_line = np.linspace(0, 1, 1000)
    w2_line = (2*Rstar - r1*w1_line)/r2
    inds = [i for i in range(len(w1_line)) if 0<=w1_line[i]<=1 and 0<=w2_line[i]<=1]
    w1_line = w1_line[inds]
    w2_line = w2_line[inds]
    if eig is not None:
        points = np.array([w1_line, w2_line]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap = ListedColormap(['black', 'red'])
        norm = BoundaryNorm([-float('inf'), 0, float('inf')], cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=1)
        lc.set_array(eig(w1_line, w2_line)) # This should work vectorized
        ax.add_collection(lc)
    else:
        ax.plot(w1_line, w2_line, color='dimgrey', linestyle='--', zorder=1)

    # Plot fixed points (if they exist)
    if model == 'additive':
        # Check for special case where additive model has an extra line of fixed points
        if alpha != 1 and np.abs(r1 + r2 - 1/(tau*(alpha - 1))) < 1e-8:
            w_line = np.linspace(0, 1, 1000)
            # For some reason linestyle doesn't work for LineCollection so do this manually
            unstable_inds = Rstar - w_line*(r1+r2)/2 >= 0
            stable_inds = Rstar - w_line*(r1+r2)/2 < 0
            ax.plot(w_line[unstable_inds], w_line[unstable_inds], color='red', linestyle=':', zorder=1)
            ax.plot(w_line[stable_inds], w_line[stable_inds], color='black', linestyle=':', zorder=1)
    elif model == 'multiplicative':
        w0 = (tau*(r1 + r2) + 1)/(tau*(1 + alpha)*(r1 + r2) + 1)
        if Rstar < w0*(r1 + r2)/2:
            ax.scatter(w0, w0, marker=marker, color='red', label='Fixed point (unstable)', 
                       zorder=10).set_clip_on(False)
        else:
            ax.scatter(w0, w0, marker=marker, color='black', label='Fixed point (stable)',
                       zorder=10).set_clip_on(False)
    elif model == 'corticostriatal':
        # Need to solve for the fixed point
        # Fixed point will always have w1 = w2 so use scalar minimization
        fun = lambda x: np.linalg.norm(
                np.array(corticostriatal_vector_field(x, x, r1, r2, Rstar, alpha, tau, T_win)),
                axis=0)
        # Other optimization methods don't respect bounds well or return one of the bounds as a
        # local minimum so use this instead
        res = sco.brute(fun, ((0.01,0.99),), Ns=1000)

        ax.scatter(res, res, marker=marker, color='black', label='Fixed point (stable)', zorder=10).set_clip_on(False)
    else:
        pass

    # Now plot any other fixed points along the boundary
    if model in model_funs:
        dw_fun = lambda x: np.array(simple_vector_field(x[0], x[1], r1, r2, Rstar, alpha, tau, model))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun, Rstar=Rstar, r=r)
    elif model == 'corticostriatal':
        dw_fun = lambda x: np.array(corticostriatal_vector_field(x[0], x[1], r1, r2, Rstar, alpha, tau, T_win))
        stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(dw_fun)
    else:
        stable_fps = []
        unstable_fps = []
        other_fps = [] 
    
    if len(stable_fps) > 0:
        ax.scatter(np.array(stable_fps)[:,0], np.array(stable_fps)[:,1], 
                   marker=marker, color='black', label='Fixed point (stable)', zorder=10).set_clip_on(False)
    if len(unstable_fps) > 0:
        ax.scatter(np.array(unstable_fps)[:,0], np.array(unstable_fps)[:,1], 
                   marker=marker, color='red', label='Fixed point (unstable)', zorder=10).set_clip_on(False)
    if len(other_fps) > 0:
        ax.scatter(np.array(other_fps)[:,0], np.array(other_fps)[:,1], 
                   marker=marker, color='green', label='Fixed point (other)', zorder=10).set_clip_on(False)

    # Let the calling function deal with axis labels, bounds, etc. because it might want to
    # plot some more stuff (like sample paths) on top


def plot_phase_plane_slow_switching(ax, model, alpha, tau, r, Rstar_vals, T_win=None):
    """
    Plot a phase portrait with task switching on a given Axes object
    r should have shape (states, N), i.e. (2, 2)
    Rstar should have shape (states,), i.e. (2,)
    For slow switching, don't plot arrows or fixed points; plot solution planes but greyed out
    """

    for cur_r, Rstar in zip(r, Rstar_vals):
        r1, r2 = cur_r[0], cur_r[1]

        if model in model_funs:
            eig = simple_eig(r1, r2, Rstar, alpha, tau, model)
        else:
            eig = None

        # Plot solution line
        w1_line = np.linspace(0, 1, 1000)
        w2_line = (2*Rstar - r1*w1_line)/r2
        if eig is not None:
            stable_w1 = []
            stable_w2 = []
            unstable_w1 = []
            unstable_w2 = []
            for i in range(len(w1_line)):
                if 0 <= w1_line[i] <= 1 and 0 <= w2_line[i] <= 1:
                    if eig(w1_line[i], w2_line[i]) <= 0:
                        stable_w1.append(w1_line[i])
                        stable_w2.append(w2_line[i])
                    else:
                        unstable_w1.append(w1_line[i])
                        unstable_w2.append(w2_line[i])
            ax.plot(stable_w1, stable_w2, color='black', label='stable', zorder=1)
            ax.plot(unstable_w1, unstable_w2, color='red', label='unstable', zorder=1)
        else:
            nonfixed_w1 = []
            nonfixed_w2 = []
            for i in range(len(w1_line)):
                if 0 <= w1_line[i] <= 1 and 0 <= w2_line[i] <= 1:
                    nonfixed_w1.append(w1_line[i])
                    nonfixed_w2.append(w2_line[i])
            ax.plot(nonfixed_w1, nonfixed_w2, color='dimgrey', linestyle='--', zorder=1)


def plot_phase_plane_fast_switching(ax, model, alpha, tau, r, Rstar_vals, T_win=None, points=10, color=None):
    """
    Plot a phase portrait with task switching on a given Axes object
    r should have shape (states, N), i.e. (2, 2)
    Rstar should have shape (states,), i.e. (2,)
    For fast switching, plot arrows for the averaged system, and numerically
    estimate the fixed point (if there is one)
    Plot solution planes, but greyed out
    """

    w1, w2 = np.meshgrid(np.linspace(0,1,points), np.linspace(0,1,points))
    w1, w2 = w1.T, w2.T

    dw1, dw2 = vector_field_fast_switching(w1, w2, r, Rstar_vals, alpha, tau, model, T_win=T_win)

    # Plot the quiver field
    if color is None:
        lengths = np.sqrt(dw1**2 + dw2**2)
        ax.quiver(w1, w2, dw1, dw2, lengths, zorder=0)
    else:
        ax.quiver(w1, w2, dw1, dw2, color=color, zorder=0)

    # Plot solution lines
    for cur_r, Rstar in zip(r, Rstar_vals):
        r1, r2 = cur_r[0], cur_r[1]
        w1_line = np.linspace(0, 1, 1000)
        w2_line = (2*Rstar - r1*w1_line)/r2
        nonfixed_w1 = []
        nonfixed_w2 = []
        for i in range(len(w1_line)):
            if 0 <= w1_line[i] <= 1 and 0 <= w2_line[i] <= 1:
                nonfixed_w1.append(w1_line[i])
                nonfixed_w2.append(w2_line[i])
        ax.plot(nonfixed_w1, nonfixed_w2, color='dimgrey', linestyle='--', zorder=1)

    # Find all fixed points
    fun = lambda x: np.array(vector_field_fast_switching(x[0], x[1], 
                            r, Rstar_vals, alpha, tau, model, T_win=T_win))
    n = 100
    vals = np.linspace(0.5/n,  1-0.5/n, n)
    results = []
    for i in range(n):
        for j in range(n):
            res = sco.root(fun, np.array([vals[i], vals[j]]))
            # Note, this excludes points on the boundaries (specifically (0,0))
            if (res.success and 1e-8 <= res.x[0] <= 1 and 1e-8 <= res.x[1] <= 1 
                    and not any(np.allclose(res2.x, res.x) for res2 in results)):
                results.append(res)
    for res in results:
        J = sco.approx_fprime(res.x, fun)
        eigs = np.linalg.eigvals(J)
        if np.all(eigs.real < 0):
            ax.scatter(res.x[0], res.x[1], marker=marker, color='black', label='Fixed point (stable)', 
                       zorder=10).set_clip_on(False)
        elif np.all(eigs.real > 0):
            ax.scatter(res.x[0], res.x[1], marker=marker, color='red', label='Fixed point (unstable)', 
                       zorder=10).set_clip_on(False)
        else:
            ax.scatter(res.x[0], res.x[1], marker=marker, color='green', label='Fixed point (other)', 
                       zorder=10).set_clip_on(False)
    
    # Now plot any other fixed points along the boundary
    # Don't filter along solution plane here
    stable_fps, unstable_fps, other_fps = get_boundary_fixed_points(fun)
    
    if len(stable_fps) > 0:
        ax.scatter(np.array(stable_fps)[:,0], np.array(stable_fps)[:,1], 
                   marker=marker, color='black', label='Fixed point (stable)', zorder=10).set_clip_on(False)
    if len(unstable_fps) > 0:
        ax.scatter(np.array(unstable_fps)[:,0], np.array(unstable_fps)[:,1], 
                   marker=marker, color='red', label='Fixed point (unstable)', zorder=10).set_clip_on(False)
    if len(other_fps) > 0:
        ax.scatter(np.array(other_fps)[:,0], np.array(other_fps)[:,1], 
                   marker=marker, color='green', label='Fixed point (other)', zorder=10).set_clip_on(False)
    # Let the calling function deal with axis labels, bounds, etc. because it might want to
    # plot some more stuff (like sample paths) on top