# generate_gp_data.py

import numpy as np
import matplotlib.pyplot as plt


# ---수정: Reference Track---
from tracks.readDataFcn import getTrack
from time2spatial import parseReference
from scipy.interpolate import make_interp_spline, CubicSpline

ref_track_ego = "mod_LMS_Track.txt"             # Ego Vehicle's Track
ref_track_obs1 = "merged_lane_track_rightmost.txt"  # Target Vehicle's Track

[Sref_ego, Xref_ego, Yref_ego, Psiref_ego, _] = getTrack(ref_track_ego)
[Sref_obs1, Xref_obs1, Yref_obs1, Psiref_obs1, _] = getTrack(ref_track_obs1)
# --- 수정 끝 ---

# ---수정: Generate spline---
def create_spline(Xref, Yref):
    x_func, y_func, _path_length, dense_s, psi, kappa = parseReference(Xref, Yref)
    x_spline = make_interp_spline(dense_s, x_func(dense_s), k=3)
    y_spline = make_interp_spline(dense_s, y_func(dense_s), k=3)
    psi_spline = make_interp_spline(dense_s, psi, k=3)
    kappa_spline = make_interp_spline(dense_s, kappa, k=3)
    return x_spline, y_spline, psi_spline, kappa_spline

x_spline_ego, y_spline_ego, psi_spline_ego, kappa_spline_ego = create_spline(Xref_ego, Yref_ego)
x_spline_obs1, y_spline_obs1, psi_spline_obs1, kappa_spline_obs1 = create_spline(Xref_obs1, Yref_obs1)
# --- 수정 끝 ---

# --- 수정: Frenet -> Cartesian 변환 함수 ---
def Frenet2Cart(frenet, x_spline, y_spline, psi_spline):
    s, n, alpha = frenet
    x_center = x_spline(s)
    y_center = y_spline(s)
    psi_center = psi_spline(s)

    dx = x_spline.derivative()(s)
    dy = y_spline.derivative()(s)
    heading = np.arctan2(dy, dx)

    n_vector = np.array([-dy, dx]) / (np.sqrt(dx**2 + dy**2) + 1e-6)

    x = x_center + n * n_vector[0]
    y = y_center + n * n_vector[1]
    psi = heading + alpha

    return x, y, psi
# --- 수정 끝 ---

#Data load
data = np.load('sim_result.npz')  #from main.py
ego_states = data['ego_state']  # (N+1, state_dim)
obs1_states = data['obs1_state']  # (N+1, state_dim)
dt = data['dt'].item() #---수정2: dt 불러오기---

#Generate Data
X_data = []
Y_data = []

Nsim = ego_states.shape[0] - 1  # number of simulation steps(2 state in 1 step)
print("Number of simulation steps:", Nsim)
for i in range(Nsim):
    ego_curr = ego_states[i]
    ego_next = ego_states[i+1]
    obs1_curr = obs1_states[i]
    obs1_next = obs1_states[i+1]

    ## --- 수정2: Frenet -> Cartesian 변환 ---
    ego_x, ego_y, ego_psi_curr = Frenet2Cart(ego_curr[:3], x_spline_ego, y_spline_ego, psi_spline_ego)
    ego_x_next, ego_y_next, ego_psi_next = Frenet2Cart(ego_next[:3], x_spline_ego, y_spline_ego, psi_spline_ego)
    obs1_x, obs1_y, obs1_psi_curr = Frenet2Cart(obs1_curr[:3], x_spline_obs1, y_spline_obs1, psi_spline_obs1)
    obs1_x_next, obs1_y_next, obs1_psi_next = Frenet2Cart(obs1_next[:3], x_spline_obs1, y_spline_obs1, psi_spline_obs1)

    
    # --- 수정2: Δ yaw 계산 ---
    ego_delta_psi = ego_psi_next - ego_psi_curr
    # 360도 고려
    if ego_delta_psi > np.pi:
        ego_delta_psi -= 2 * np.pi
    elif ego_delta_psi < -np.pi:
        ego_delta_psi += 2 * np.pi
    #---수정2 끝---
    
    # --- 수정2: Δ yaw 계산 ---
    obs1_delta_psi = obs1_psi_next - obs1_psi_curr
    # 360도 고려
    if obs1_delta_psi > np.pi:
        obs1_delta_psi -= 2 * np.pi
    elif obs1_delta_psi < -np.pi:
        obs1_delta_psi += 2 * np.pi
    #---수정2 끝---


    delta_x = obs1_x - ego_x
    delta_y = obs1_y - ego_y

    # Δs, Δey 계산 (ego 경로 기준 방향으로 투영)
    psi_ref = psi_spline_ego(ego_curr[0])  # ego가 달리는 트랙 방향
    cos_psi = np.cos(psi_ref)
    sin_psi = np.sin(psi_ref)

    delta_s = delta_x * cos_psi + delta_y * sin_psi
    delta_ey = -delta_x * sin_psi + delta_y * cos_psi

    # --- 수정: Input X ---
    s_ego = ego_curr[0]              # ego s
    s_obs1 = obs1_curr[0]            # obs1 s

    ey_ego = ego_curr[1]             # ego lateral offset
    ey_tv = obs1_curr[1]             # obs1 lateral offset

    ephi_ego = ego_curr[2]           # ego heading error
    ephi_tv = obs1_curr[2]           # obs1 heading error

    vx_ego = ego_curr[3]             # ego long velocity
    vx_tv = obs1_curr[3]             # obs1 long velocity    
    vy_ego = (ego_y_next-ego_y)/dt   # ego trans velocity
    vy_tv = (obs1_y_next-obs1_y)/dt
    a_tran_ego = ego_curr[-1]            # ego tran acceleration
    a_tran_tv = obs1_curr[-1]            # obs1 tran velocity
    a_long_ego = ego_curr[-2]        # ego long acceleration
    a_long_tv = obs1_curr[-2]        # obs1 long acceleration
    
            
    pedal_ego = ego_curr[4]          # ego pedal
    pedal_tv = obs1_curr[4]          # obs1 pedal

    steer_ego = ego_curr[5]          # ego steering
    steer_tv = obs1_curr[5]          # obs1 steering

    omega_ego = ego_curr[-1]     
    omega_tv = obs1_curr[-1]  

    delta_s_tv = obs1_next[0] - ego_next[0]     # delta_s
    delta_ey_tv = obs1_next[1] - ego_next[1]    # delta_ey
    delta_psi_tv = obs1_next[2] - ego_next[2]   # delta_psi
    delta_vx_tv = obs1_next[3] - obs1_curr[3]   # delta_vx
    delta_vy_tv = obs1_next[-2] - obs1_curr[-2] # delta_vy
    delta_w_tv = obs1_next[-1] - obs1_curr[-1]  # delta_w

    kappa = kappa_spline_obs1(obs1_curr[0])  # obs1 위치에서 트랙 곡률

    X = np.array([
        dt * i,        # time

        ego_x,         # global position
        ego_y,
        obs1_x,
        obs1_y,
        
        vx_ego,        # BodyLinearVelocity
        vx_tv,
        vy_ego,
        vy_tv,
        omega_ego,
        omega_tv,
        a_long_ego,
        a_long_tv,
        a_tran_ego,
        a_tran_tv,

        ego_psi_curr,  # global orientation
        obs1_psi_curr, 

        s_ego,         # parametric
        s_obs1,
        ey_ego,
        ey_tv,
        ephi_ego,
        ephi_tv,

        pedal_ego,     # actuation
        pedal_tv,
        steer_ego,
        steer_tv,

        kappa          # curvature
    ])
    X_data.append(X)

    # --- Output Y ---
    Y = np.array([
        delta_s_tv,
        delta_ey_tv,
        delta_psi_tv,
        delta_vx_tv,
        delta_vy_tv,
        delta_w_tv,
    ])
    Y_data.append(Y)

#Save
np.savez('gp_training_data.npz', X=np.array(X_data), Y=np.array(Y_data))

print("GP Training Data Saved")
print("X shape:", np.array(X_data).shape, "Y shape:", np.array(Y_data).shape)