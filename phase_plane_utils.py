import numpy as np
import scipy.optimize as sco

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


def get_boundary_fixed_points(dx_fun, fps_to_filter=None, points=100, eps=1e-8):
    """
    Get fixed points (including artificial ones caused by clipping) along the boundary of [0,1]^2
    dx_fun should take in x=[x1, x2] and return dx=[dx1, dx2], with vectorization in the 2nd dimension
    """

    stable_fps = []
    unstable_fps = []
    other_fps = []
    
    # First examine boundaries, away from corners
    t_vals = np.arange(1/(points+1), 1-1/(points+2), 1/(points+1)).reshape(1,-1)
    for offset, tangent, inward_normal in [(np.array([0,0]), np.array([1,0]), np.array([0,1])),
                                           (np.array([0,0]), np.array([0,1]), np.array([1,0])),
                                           (np.array([0,1]), np.array([1,0]), np.array([0,-1])),
                                           (np.array([1,0]), np.array([0,1]), np.array([-1,0]))]:
        offset = offset.reshape(2,1)
        tangent = tangent.reshape(2,1)
        inward_normal = inward_normal.reshape(2,1)
        # We want points where dw_tangent = 0 and dw_inwards <= 0
        dw_inwards = lambda t: np.sum(dx_fun(offset + t*tangent)*inward_normal, axis=0)
        dw_tangent = lambda t: np.sum(dx_fun(offset + t*tangent)*tangent, axis=0)
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
            inwards = np.sum(dx_fun(offset + t*tangent + eps*inward_normal)*inward_normal, axis=0)
            if pos_side < 0 and neg_side > 0 and inwards < 0:
                stable_fps.append(offset.flatten() + t*tangent.flatten())
            elif pos_side > 0 and neg_side < 0 and inwards > 0:
                unstable_fps.append(offset.flatten() + t*tangent.flatten())
            else:
                other_fps.append(offset.flatten() + t*tangent.flatten())
    # Now check corners
    for corner, ax1, ax2 in [(np.array([0,0]), np.array([1,0]), np.array([0,1])),
                             (np.array([1,0]), np.array([-1,0]), np.array([0,1])),
                             (np.array([1,1]), np.array([-1,0]), np.array([0,-1])),
                             (np.array([0,1]), np.array([1,0]), np.array([0,-1]))]:
        dw = np.linalg.norm(dx_fun(corner))
        dw_ax1 = np.dot(dx_fun(corner + eps*ax1), ax1)
        dw_ax2 = np.dot(dx_fun(corner + eps*ax2), ax2)
        if dw_ax1 < 0 and dw_ax2 < 0:
            stable_fps.append(corner)
        elif dw < eps and dw_ax1 > 0 and dw_ax2 > 0:
            unstable_fps.append(corner)
        elif dw < eps:
            other_fps.append(corner)

    if fps_to_filter is not None:
        stable_fps = [fp for fp in stable_fps if not any(
            np.allclose(fp, fp_to_filter, atol=eps) for fp_to_filter in fps_to_filter)]
        unstable_fps = [fp for fp in unstable_fps if not any(
            np.allclose(fp, fp_to_filter, atol=eps) for fp_to_filter in fps_to_filter)]
        other_fps = [fp for fp in other_fps if not any(np.allclose(
            fp, fp_to_filter, atol=eps) for fp_to_filter in fps_to_filter)]
        
    return stable_fps, unstable_fps, other_fps