from tracks.readDataFcn import getTrack
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline,CubicSpline
import scipy.interpolate as interp
import math

def get_spline(Xref, Yref):
    x_func, y_func, _path_length, dense_s, psi, kappa = parseReference(Xref, Yref)

    x_list = x_func(dense_s)
    y_list = y_func(dense_s)

    x_spline     = make_interp_spline(dense_s, x_list, k=3)
    y_spline     = make_interp_spline(dense_s, y_list, k=3)
    psi_spline   = make_interp_spline(dense_s, psi,    k=3)
    kappa_spline = make_interp_spline(dense_s, kappa,  k=3)

    spl_x     = x_spline.c
    spl_y     = y_spline.c
    spl_psi   = psi_spline.c
    spl_kappa = kappa_spline.c
    spl_knots = kappa_spline.t

    # kapparef_s = Function.bspline('kapparef_s', [spl_knots ], spl_kappa, [3], 1)
    x     = CubicSpline(dense_s, spl_x,     bc_type='clamped')
    y     = CubicSpline(dense_s, spl_y,     bc_type='clamped')
    kappa = CubicSpline(dense_s, spl_kappa, bc_type='clamped')
    psi   = CubicSpline(dense_s, spl_psi,   bc_type='clamped')
    
    num_points = int(_path_length*100)

    s = np.linspace(0, dense_s[-1]-1, num_points)

    # path_points = np.column_stack((x_spline(s_vals), y_spline(s_vals)))

    return x_spline(s), y_spline(s), s, x, y, kappa, psi, dense_s

class Ref:
    x:np.array            = None
    y:np.array            = None
    s:np.array            = None
    s_list:np.array       = None
    x_spl:CubicSpline     = None
    y_spl:CubicSpline     = None
    kappa_spl:CubicSpline = None
    psi_spl:CubicSpline   = None

def parseReference(x, y):
    
    if len(x) < 2:
        print("Reference path must have at least 2 points")
        return None, None, None, None, None, None
    
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    dist_s = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    xe, xee = x[-1], x[-2]
    ye, yee = y[-1], y[-2]
    dist_e  = ((xe - xee) ** 2 + (ye - yee) ** 2) ** 0.5

    x_slope_start, x_slope_end = (x2 - x1) / dist_s, (xe - xee) / dist_e
    y_slope_start, y_slope_end = (y2 - y1) / dist_s, (ye - yee) / dist_e
    
    ds = [0]
    distance = 0
    
    for i in range(1, len(x)):
        
        x_now, y_now = x[i], y[i]
        x_prv, y_prv = x[i-1], y[i-1]
        distance = math.sqrt((x_now - x_prv)**2 + (y_now - y_prv)**2)
        ds.append(distance + ds[-1])

    x_spline = interp.CubicSpline(ds, x, bc_type=((1, x_slope_start), (1, x_slope_end)))
    y_spline = interp.CubicSpline(ds, y, bc_type=((1, y_slope_start), (1, y_slope_end)))

    density = 0.05
    dense_s = np.linspace(ds[0], ds[-1], int(ds[-1]/density)) # Change 1000 to the density you want

    # Get first derivatives
    dx_ds = x_spline.derivative()(dense_s)
    dy_ds = y_spline.derivative()(dense_s)

    # Get second derivatives
    dx2_ds2 = x_spline.derivative(nu=2)(dense_s)
    dy2_ds2 = y_spline.derivative(nu=2)(dense_s)

    # Compute phi (slope angle)
    phi = np.arctan2(dy_ds, dx_ds)

    # Compute kappa (curvature)
    kappa = (dx_ds * dy2_ds2 - dy_ds * dx2_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)

    return x_spline, y_spline, ds[-1], dense_s, phi, kappa

ref_track         = "mod_LMS_Track.txt"
merged_track_file = "merged_lane_track_rightmost.txt"
second_track_file = "shifted_lane_track.txt"

[Sref,      Xref,      Yref,      Psiref,      _] = getTrack(ref_track)
[Sref_obs1, Xref_obs1, Yref_obs1, Psiref_obs1, _] = getTrack(merged_track_file)
[Sref_obs2, Xref_obs2, Yref_obs2, Psiref_obs2, _] = getTrack(second_track_file)

ref_ego = Ref()
ref_obs1 = Ref()
ref_obs2 = Ref()

ref_ego.x,  ref_ego.y,  ref_ego.s,  ref_ego.x_spl,  ref_ego.y_spl,  ref_ego.kappa_spl,  ref_ego.psi_spl,  ref_ego.s_list  = get_spline(Xref, Yref)
ref_obs1.x, ref_obs1.y, ref_obs1.s, ref_obs1.x_spl, ref_obs1.y_spl, ref_obs1.kappa_spl, ref_obs1.psi_spl, ref_obs1.s_list = get_spline(Xref_obs1, Yref_obs1)
ref_obs2.x, ref_obs2.y, ref_obs2.s, ref_obs2.x_spl, ref_obs2.y_spl, ref_obs2.kappa_spl, ref_obs2.psi_spl, ref_obs2.s_list = get_spline(Xref_obs2, Yref_obs2)

lane_width = 0.24  # Lane width, modify as needed

# Setup plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_ylim(bottom=-1.75 * 5, top=0.35 * 5)
ax.set_xlim(left=-1.1 * 5, right=1.6 * 5)
ax.set_ylabel('y[m]')
ax.set_xlabel('x[m]')

# Draw center line (3rd lane center)
ax.plot(Xref, Yref, '--', color='gray')

# Draw lanes' center lines
lanes = [-2, -1, 0, 1]  # Offset for each lane center from the center line
for lane in lanes:
    if lane != 0:
        X_center = Xref + lane * lane_width * np.sin(Psiref)
        Y_center = Yref - lane * lane_width * np.cos(Psiref)
        ax.plot(X_center, Y_center, '--', color='gray', linewidth=1)

# Draw lane boundaries
boundaries = [-2.5, -1.5, -0.5, 0.5, 1.5]  # Offset for each lane boundary from the center line
for boundary in boundaries:
    Xbound = Xref + boundary * lane_width * np.sin(Psiref)
    Ybound = Yref - boundary * lane_width * np.cos(Psiref)
    ax.plot(Xbound, Ybound, color='k', linewidth=1)

plt.show()