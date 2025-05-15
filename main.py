import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from acados_settings_dev import *
from plotFcn import *
from tracks.readDataFcn import getTrack
from time2spatial import *
from casadi import *
from scipy.interpolate import make_interp_spline,CubicSpline
from scipy.spatial.distance import cdist

from scipy.integrate import solve_ivp

from matplotlib.animation import FFMpegWriter

"""
Example of the frc_racecars in simulation without obstacle avoidance:
This example is for the optimal racing of the frc race cars. The model is a simple bicycle model and the lateral acceleration is constraint in order to validate the model assumptions.
The simulation starts at s=-2m until one round is completed(s=8.71m). The beginning is cut in the final plots to simulate a 'warm start'.
"""

class Car:
    s:float        = 0.0
    n:float        = 0.0
    alpha:float    = 0.0
    x:float        = 0.0
    y:float        = 0.0
    psi:float      = 0.0
    v:float        = 0.0
    delta:float    = 0.0
    pedal:float    = 0.0
    dpedal:float   = 0.0
    ddelta:float   = 0.0
    l:float        = 0.15
    w:float        = 0.075
    cart:np.array  = None
    state:np.array = None
    input:np.array = None
    

class Ref:
    x:np.array            = None
    y:np.array            = None
    s:np.array            = None
    s_list:np.array       = None
    x_spl:CubicSpline     = None
    y_spl:CubicSpline     = None
    kappa_spl:CubicSpline = None
    psi_spl:CubicSpline   = None


def Cart2Frenet(cart, ref):
    # 경로 상의 점들과 월드 좌표계 상의 차량 위치 간의 거리 계산
    dist = cdist(np.column_stack((ref.x, ref.y)), [[cart[0], cart[1]]])
    c_idx = np.argmin(dist)
    c_xy = np.append(ref.x[c_idx], ref.y[c_idx])
    c_s = ref.s[c_idx]

    # 가장 가까운 s 값을 찾은 후, 프레네 좌표계에서 d 값을 계산
    # tmp_psi = self.psi_spline(closest_s)
    vx = ref.x_spl.derivative()(c_s)
    vy = ref.y_spl.derivative()(c_s)
    yaw = np.arctan2(vy, vx)

    # d는 월드 좌표계에서 프레네 경로와의 수직 거리
    dx = cart[0] - c_xy[0]
    dy = cart[1] - c_xy[1]
    d = np.sqrt(dx**2 + dy**2)
    n = np.array([-vy, vx])
    xy = np.array([dx, dy])
    d *= np.sign(np.dot(xy, n))  # Adjust sign based on the side of the path
    
    # yaw_s는 프레네 경로 상의 yaw (tmp_psi)와 월드 좌표계의 yaw 차이, normalized
    alpha = (ref.psi_spl(c_s) - yaw)
    if alpha > np.pi:
        alpha -= 2*np.pi
    elif alpha < -np.pi:
        alpha += 2*np.pi

    return c_s, d, alpha


def Frenet2Cart(frenet, ref):
    """
    Convert a point from Frenet coordinates to Cartesian coordinates.

    Args:
        s (float): Longitudinal coordinate along the path.
        d (float): Lateral deviation from the path.
        yaw_s (float): Orientation in the Frenet frame.

    Returns:
        tuple: (x, y, yaw), the Cartesian coordinates and orientation.
    """
    # Evaluate the splines at the given s
    x_center = ref.x_spl(frenet[0])
    y_center = ref.y_spl(frenet[0])

    dx = ref.x_spl.derivative()(frenet[0])
    dy = ref.y_spl.derivative()(frenet[0])
    psi = np.arctan2(dy, dx)
    n = np.array([-dy, dx]) / (np.sqrt(dx**2 + dy**2) + 1e-6)

    # Compute Cartesian coordinates
    x = x_center + frenet[1] * n[0]
    y = y_center + frenet[1] * n[1]
    psi = psi + frenet[2]
    
    return x, y, psi

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

def get_rel(ego, obs, ref):
    """
    ego를 기준으로 obstacle의 프레네 좌표 상대 위치(s, n)를 계산합니다.
    
    Args:
        ego: 기준 차량 객체 (ego)
        obstacle: 상대 차량 객체 (obs1 또는 obs2)
        ref: 프레네 좌표 변환에 사용할 기준 레퍼런스 (예: ref_ego)
    
    Returns:
        dict: {'s_rel': 상대 s 값, 'n_rel': 상대 n 값}
    """
    # ego의 프레네 좌표 계산
    ego_s, ego_d, _ = Cart2Frenet(ego, ref)
    # obstacle의 프레네 좌표 계산
    obs_s, obs_d, _ = Cart2Frenet(obs, ref)
    
    # ego 기준의 상대 프레네 좌표 계산
    rel_s = obs_s - ego_s
    rel_n = obs_d - ego_d
    
    return [rel_s, rel_n]

def dist2obs(rel, l, w, degree=2):
    s_rot = (+(rel[0])*cos(rel[2]) + (rel[1])*sin(rel[2])) * (w)
    n_rot = (-(rel[0])*sin(rel[2]) + (rel[1])*cos(rel[2])) * (l)
    dist = (s_rot**degree + n_rot**degree)**(1/degree)
    return dist

def _dynamics_of_car(x0) -> list:
    """
    Used for forward propagation. This function takes the dynamics from the acados model.
    """
    ## Race car parameters
    m = 0.043
    C1 = 0.5
    C2 = 15.5
    Cm1 = 0.28
    Cm2 = 0.05
    Cr0 = 0.011
    Cr2 = 0.006

    s, n, alpha, v, pedal, delta, derD, derDelta = x0

    # dynamics
    Fxd = (Cm1 - Cm2 * v) * pedal - Cr2 * v * v - Cr0 * tanh(5 * v)
    sdot = (v * cos(alpha + C1 * delta)) / (1 - ref_ego.kappa_spl(s) * n)
    ndot = v * sin(alpha + C1 * delta)
    alphadot = v * C2 * delta - ref_ego.kappa_spl(s) * sdot
    vdot = Fxd / m * cos(C1 * delta)
    Ddot = derD
    deltadot = derDelta


    xdot = [float(sdot), ndot, float(alphadot), vdot, Ddot, deltadot, Ddot, deltadot]

    return xdot




# Main
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

# Load models
Tf = 1  # prediction horizon
N = 10  # number of discretization steps
dt = Tf / N
T = 15.00  # maximum simulation time[s]
target_s = 5  # reference for final reference progress

ego_constraint,  ego_model,  ego_solver,  ego_ocp  = acados_settings(Tf, N, ref_track, model_name="EgoModel")
obs1_constraint, obs1_model, obs1_solver, obs1_ocp = acados_setting(Tf, N, merged_track_file, model_name="Obs1Model")
obs2_constraint, obs2_model, obs2_solver, obs2_ocp = acados_setting(Tf, N, second_track_file, model_name="Obs2Model")

# Dimensions
nx = ego_model.x.rows()
nu = ego_model.u.rows()
ny = nx + nu
Nsim = int(T * N / Tf)

# Initialize data structs
objs = 3
simX = np.zeros((objs, Nsim, nx))
simU = np.zeros((objs, Nsim, nu))
x_next = np.zeros((objs, N, nx))
x_pred = np.zeros(())


ego = Car()
obs1 = Car()
obs2 = Car()

# --- 수정: Trajectory 기록용 리스트 추가 ---
ego_traj = []   # 실제 ego 차량 주행 traj 기록
obs1_traj = []  # 실제 obs1 차량 주행 traj 기록
# --- 수정 끝 ---

#Data Generation -> random state function
if 'SIM_SEED' in os.environ:
    seed = int(os.environ['SIM_SEED'])
    np.random.seed(seed)
else:
    seed = 0
    np.random.seed(seed)

# 랜덤 초기조건 설정
ego.s = np.random.uniform(0.5, 1.5)
ego.n = np.random.uniform(-0.1, 0.1)
ego.v = np.random.uniform(2.5, 4.0)

obs1.s = np.random.uniform(1.0, 2.0)
obs1.n = np.random.uniform(-0.1, 0.1)
obs1.v = np.random.uniform(2.5, 4.0)

obs1.y, obs2.y = -0.24, 0.24
ego.state, obs1.state, obs2.state = np.zeros((N+1, nx)), np.zeros((N+1, nx)), np.zeros((N+1, nx))
ego.input, obs1.input, obs2.input = np.zeros((N+1, nu)), np.zeros((N+1, nu)), np.zeros((N+1, nu))
ego.cart, obs1.cart, obs2.cart    = np.zeros((N+1, 4)), np.zeros((N+1, 4)), np.zeros((N+1, 4))

# 먼저, 객체와 대응하는 solver를 리스트로 정의합니다.
cars    = [ego, obs1, obs2]
solvers = [ego_solver, obs1_solver, obs2_solver]

tcomp_sum = 0
tcomp_max = 0

# Carlength = 0.15
# Carwidth = 0.075

Carlength = 0.15
Carwidth = 0.075

# Plot ego vehicle
ego_rect = plt.Rectangle((0, 0), Carlength, Carwidth, fc='green', ec='black')
ax.add_patch(ego_rect)

# Plot obs1 vehicle
obs1_rect = plt.Rectangle((0, 0), Carlength, Carwidth, fc='red', ec='black')
ax.add_patch(obs1_rect)

# Plot obs2 vehicle
obs2_rect = plt.Rectangle((0, 0), Carlength, Carwidth, fc='blue', ec='black')
ax.add_patch(obs2_rect)

# Show the plot
# plt.ion()
# plt.show(block=False)
# plt.pause(0.1)

# Simulation
for i in range(Nsim):
    # Update reference for ego
    for car, solver in zip(cars, solvers):
        s_ref = car.s + target_s
        for j in range(N):
            yref = np.array([car.s + (target_s) * (j+1) / N, 0, 0, 0, 0, 0, 0, 0])
            solver.set(j, "yref", yref)
        yref_N = np.array([s_ref, 0, 0, 0, 0, 0])
        solver.set(N, "yref", yref_N)

    # get relative state from ego to obs
    # ego2obs1 = get_rel(ego, obs1, ref_ego)
    # ego2obs2 = get_rel(ego, obs2, ref_ego)

    # QP - find max h
    

    # find hdot
    
    # ACADOS parameter update

    for step in range(N+1):
        obs1.cart[step][0], obs1.cart[step][1], obs1.cart[step][2] = Frenet2Cart(obs1.state[step], ref_obs1)
        obs2.cart[step][0], obs2.cart[step][1], obs2.cart[step][2] = Frenet2Cart(obs2.state[step], ref_obs2)
    
    params = np.zeros((N, 6*(objs-1)+1))
    for step in range(N):
        obs1_p = np.append(obs1.cart[step+1, :4], [0.0, 1.0])
        obs2_p = np.append(obs2.cart[step+1, :4], [0.0, 1.0])
        params[step] = np.concatenate((obs1_p, obs2_p, [0.0]))
    # print(params[0])
    # print("#"*50)
    # print("#"*50)
    # Solve ego ocp
    t = time.time()
    for step in range(N):
        ego_solver.set(step, "p", params[step])
    #     obs1_acados_solver.set(step, "p", Obs1sObs)
    #     obs2_acados_solver.set(step, "p", Obs2sObs)

    status = ego_solver.solve()
    status_obs1 = obs1_solver.solve()
    status_obs2 = obs2_solver.solve()
    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - t

    # Manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    for car, solver in zip(cars, solvers):
        for step in range(N + 1):
            # Get solution for ego
            car.state[step] = solver.get(step, field_="x")
            car.cart[step] = solver.get(step, field_="x")[:4]
            if step == N:
                break
            car.input[step] = solver.get(step, field_="u")
    state = np.concatenate((ego.state[0], ego.input[0]))
    
    # RK4
    solution = solve_ivp(
            lambda t, x: _dynamics_of_car(x),
            t_span=[0, dt],
            y0=state,
            # args=(np.clip(ego.input[0], ego_ocp.constraints.lbu, ego_ocp.constraints.ubu),),
            method="RK45",
            atol=1e-8,
            rtol=1e-8,
        )
    solution = [x[-1] for x in solution.y]
    ego.s, ego.n, ego.alpha, ego.v, ego.pedal, ego.delta = solution[0], solution[1], solution[2], solution[3], solution[4], solution[5]
    # ego.s, ego.n, ego.alpha, ego.v, ego.pedal, ego.delta = ego.state[1][0], ego.state[1][1], ego.state[1][2], ego.state[1][3], ego.state[1][4], ego.state[1][5]
    obs1.s, obs1.n, obs1.alpha, obs1.v, obs1.pedal, obs1.delta = obs1.state[1][0], obs1.state[1][1], obs1.state[1][2], obs1.state[1][3], obs1.state[1][4], obs1.state[1][5]
    obs2.s, obs2.n, obs2.alpha, obs2.v, obs2.pedal, obs2.delta = obs2.state[1][0], obs2.state[1][1], obs2.state[1][2], obs2.state[1][3], obs2.state[1][4], obs2.state[1][5]
    
    #---수정 종횡가속도추가---
    m = 0.043
    C1 = 0.5
    C2 = 15.5
    Cm1 = 0.28
    Cm2 = 0.05
    Cr0 = 0.011
    Cr2 = 0.006

    # Ego accelerations
    Fxd_ego = (Cm1 - Cm2 * ego.v) * ego.pedal - Cr2 * ego.v**2 - Cr0 * np.tanh(5 * ego.v)
    a_long_ego = Fxd_ego / m
    a_tran_ego = C2 * ego.v**2 * ego.delta + Fxd_ego * np.sin(C1 * ego.delta) / m

    # Obs1 accelerations
    Fxd_obs1 = (Cm1 - Cm2 * obs1.v) * obs1.pedal - Cr2 * obs1.v**2 - Cr0 * np.tanh(5 * obs1.v)
    a_long_obs1 = Fxd_obs1 / m
    a_tran_obs1 = C2 * obs1.v**2 * obs1.delta + Fxd_obs1 * np.sin(C1 * obs1.delta) / m

    # Obs1 transverse velocity
    v_trans_ego = ego.v * np.sin(ego.alpha + C1 * ego.delta)
    v_trans_obs1 = obs1.v * np.sin(obs1.alpha + C1 * obs1.delta)


    # Ego Obs1 yaw rate
    kappa_ego = ref_ego.kappa_spl(ego.s)
    sdot_ego = ego.v * np.cos(ego.alpha + C1 * ego.delta) / (1 - kappa_ego * ego.n)
    alphadot_ego = ego.v * C2 * ego.delta - kappa_ego * sdot_ego
    yaw_rate_ego = alphadot_ego + kappa_ego * sdot_ego

    kappa_obs1 = ref_obs1.kappa_spl(obs1.s)
    sdot_obs1 = obs1.v * np.cos(obs1.alpha + C1 * obs1.delta) / (1 - kappa_obs1 * obs1.n)
    alphadot_obs1 = obs1.v * C2 * obs1.delta - kappa_obs1 * sdot_obs1
    yaw_rate_obs1 = alphadot_obs1 + kappa_obs1 * sdot_obs1
    #---수정끝---

    x0 = solution[:6]
    u0 = solution[6:]

    # for j in range(nx):
    #     simX[i, j] = x0[j]
    # for j in range(nu):
    #     simU[i, j] = u0[j]

    # 1. 초기 조건 업데이트 (MPC solver에 실제 상태를 넣습니다)
    ego_solver.set(0, "lbx", np.array(x0))
    ego_solver.set(0, "ubx", np.array(x0))
    s0 = ego.state[1]  # 실제 상태의 s 성분

    # Update initial condition for obs1 using its current state
    obs1_solver.set(0, "lbx", obs1.state[1])
    obs1_solver.set(0, "ubx", obs1.state[1])
    s0_obs1 = obs1.state[1][0]

    # Update initial condition for obs2 using its current state
    obs2_solver.set(0, "lbx", obs2.state[1])
    obs2_solver.set(0, "ubx", obs2.state[1])
    s0_obs2 = obs2.state[1][0]

    # Convert current Frenet state to Cartesian coordinates for plotting.
    # 각 차량의 참조 객체(ref_ego, ref_obs1, ref_obs2)는 해당 경로 정보를 담고 있어야 합니다.
    ego.x,  ego.y,   ego.psi = Frenet2Cart(ego.state[1],  ref_ego)
    obs1.x, obs1.y, obs1.psi = Frenet2Cart(obs1.state[1], ref_obs1)
    obs2.x, obs2.y, obs2.psi = Frenet2Cart(obs2.state[1], ref_obs2)


    # --- 수정: 현재 주행한 상태를 기록 (curr state) ---
    ego_traj.append([ego.s, ego.n, ego.alpha, ego.v, ego.pedal, ego.delta, a_long_ego, a_tran_ego, v_trans_ego, yaw_rate_ego])
    obs1_traj.append([obs1.s, obs1.n, obs1.alpha, obs1.v, obs1.pedal, obs1.delta, a_long_obs1, a_tran_obs1, v_trans_obs1, yaw_rate_obs1])
    # --- 수정 끝 ---


    # Update vehicle plots using the computed Cartesian coordinates.
    update_vehicle_plot(ego_rect,  ego.x,  ego.y,  ego.psi, Carlength, Carwidth)
    update_vehicle_plot(obs1_rect, obs1.x, obs1.y, obs1.psi, Carlength, Carwidth)
    update_vehicle_plot(obs2_rect, obs2.x, obs2.y, obs2.psi, Carlength, Carwidth)

    plt.draw()
    plt.pause(0.001)  # Add a small pause to update the plot in real time

# Average speed 계산 (Ego 차량 기준)
avg_speed = np.mean([state[3] for state in ego.state[:-1]])

# Print some stats
print("Average computation time: {}".format(tcomp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))
#print("Average speed:{}m/s".format(np.average(simX[:, 3])))
print("Average speed: {:.2f} m/s".format(avg_speed))
print("Lap time: {}s".format(Tf * Nsim / N))

# Keep the plot open after the simulation
plt.ioff()
plt.show()

# --- 수정: Trajectory 기록 저장 ---
ego_traj = np.array(ego_traj)
obs1_traj = np.array(obs1_traj)
plt.close('all')
# ---수정2: dt도 함께 저장---
np.savez('sim_result.npz', ego_state=ego_traj, obs1_state=obs1_traj, dt=Tf/N)
# --- 수정 끝 ---
print(ego_traj[0][-1], ego_traj[0][-2])
print(ego_traj[10][-1], ego_traj[10][-2])