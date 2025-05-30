import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import json

unc_mode = input("input uncertainty mode(get/load/none): ")
NOISE_FILE = 'uncertainties.json'
# prepare uncertainties
if unc_mode=='get' : 
    np.random.seed(0)
    noises=[]
elif unc_mode=='load':
    noises=json.load(open(NOISE_FILE,'r'))
else:
    noises=None

# System parameters
dt = 0.05  # Time step
T = 30.0  # Extended simulation time
N = int(T / dt)  # Number of time steps
alpha = 0.8  # Reduced CBF tuning parameter
gamma = 0.5  # CLF tuning parameter for faster convergence
slack_weight = 1e3 # Slack weight for CBF

# Robot initial position, desired velocity, and goal
p = np.array([0.0, 0.0])  # Initial position [x, y]
u_des = np.array([1.0, 1.0])  # Desired velocity [u_x, u_y]
p_goal = np.array([4.0, 4.0])  # Goal position

# Obstacle parameters
p_o = np.array([2.0, 2.0])  # Obstacle center
r = 0.5  # Obstacle radius
d = 0.25  # safety margin

# Check initial CBF condition
h_initial = np.linalg.norm(p - p_o) - r
if h_initial < 0:
    print(f"Initial position inside obstacle: h(p) = {h_initial:.4f}")
    exit()

# Store trajectory for plotting
positions = [p.copy()]

# Set up CasADi Opti stack
opti = ca.Opti()

# Define variables
u_1 = opti.variable(2)       # Control input [u_x, u_y]
u_2 = opti.variable()        # Slack derivative
sc = opti.variable()         # Slack variable for CBF
s = opti.variable()          # Slack variable for CLF
p_sym = opti.parameter(2)    # Position
p_o_sym = opti.parameter(2)  # Obstacle center
r_sym = opti.parameter()     # Obstacle radius
d_sym = opti.parameter()         # safety margin
u_des_sym = opti.parameter(2)  # Desired velocity
p_goal_sym = opti.parameter(2)  # Goal position
alpha_sym = opti.parameter()  # CBF alpha
gamma_sym = opti.parameter()  # CLF gamma
slack_weight_sym = opti.parameter()  # Slack weight

# Define CBF: h(p) = ||p - p_o|| - r
h = ca.sqrt(ca.dot(p_sym - p_o_sym, p_sym - p_o_sym)) - (r_sym + d_sym)
dh_dp = (p_sym - p_o_sym) / ca.sqrt(ca.dot(p_sym - p_o_sym, p_sym - p_o_sym))  # Gradient: (p - p_o)/||p - p_o||

# Define CLF: V(p) = 0.5 * ||p - p_goal||^2
V = 0.5 * ca.dot(p_sym - p_goal_sym, p_sym - p_goal_sym) + 0.5 * ca.dot(u_1, u_1) + 0.5 * ca.dot(u_2, u_2)
dV_dp = p_sym - p_goal_sym + u_1  # Gradient: p - p_goal

sc_ref = 0.125
# Define QP cost: 0.5 * ||u - u_des||^2 + slack_weight * s^2
cost = 0.5 * ca.dot(u_1 - u_des_sym, u_1 - u_des_sym) + slack_weight_sym * s**2 + slack_weight_sym * sc**2 #+ 0.5 * ca.dot(sc-sc_ref, sc-sc_ref)

# Define constraints
cbf_constraint = ca.dot(dh_dp, u_1) + u_2 >= -alpha_sym * (h + sc)  # CBF with slack
clf_constraint = ca.dot(dV_dp, u_1) <= -gamma_sym * V + s # CLF
clf_slack_constraint = s >= 0  # Non-negative slack
cbf_slack_constraint = sc >= 0
cbf_slack_constraint_upper_bound = sc <= d_sym

# Set up QP
opti.minimize(cost)
opti.subject_to(cbf_constraint)
opti.subject_to(clf_constraint)
opti.subject_to(clf_slack_constraint)
opti.subject_to(cbf_slack_constraint)
opti.subject_to(cbf_slack_constraint_upper_bound)

# Optimize solver settings
opti.solver('ipopt', {'print_time': False}, {'max_iter': 100, 'tol': 1e-6})

slack_count = 0
slack_values = []

# Simulation loop
slack_count = 0
for k in range(N):

    if unc_mode=='get': 
        noise=np.random.randn(2)*0.01; noises.append(noise.tolist())
    elif unc_mode=='load': 
        noise=np.array(noises[k])
    else:                   
        noise=np.zeros(2)


    # Set parameter values
    opti.set_value(p_sym, p)
    opti.set_value(p_o_sym, p_o)
    opti.set_value(r_sym, r)
    opti.set_value(d_sym, d)
    opti.set_value(u_des_sym, u_des)
    opti.set_value(p_goal_sym, p_goal)
    opti.set_value(alpha_sym, alpha)
    opti.set_value(gamma_sym, gamma)
    opti.set_value(slack_weight_sym, slack_weight)

    # Solve QP
    try:
        sol = opti.solve()
        u_val = sol.value(u_1)
        dsc_val = sol.value(u_2)
        s_val = sol.value(sc)
        if s_val > 1e-3:
            print(f"Step {k+1}: CBF slack used, s = {s_val:.4f}")
            slack_count += 1
            slack_values.append(s_val)
    except Exception as e:
        print(f"QP infeasible at step {k+1}: {e}")
        break
    
    # Update position
    p += u_val * dt + noise
    sc += dsc_val *dt
    positions.append(p.copy())

    # Check if goal is reached
    if np.linalg.norm(p - p_goal) < 0.1:
        print(f"Goal reached at step {k+1}: p = {p}")
        positions = positions[:k+2]
        break

    
if unc_mode=='get': 
        json.dump(noises,open(NOISE_FILE,'w'))

# Convert positions to array
positions = np.array(positions)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Robot trajectory')
plt.plot(p_o[0], p_o[1], 'ro', label='Obstacle center')
plt.plot(p_goal[0], p_goal[1], 'g*', label='Goal', markersize=15)
theta = np.linspace(0, 2*np.pi, 100)
circle_x = p_o[0] + r * np.cos(theta)
circle_y = p_o[1] + r * np.sin(theta)
plt.plot(circle_x, circle_y, 'r--', label='Obstacle boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Obstacle Avoidance with Square Root CBF and CLF (CasADi)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('obstacle_avoidance_sqrt_cbf_with_uncertainty.png', bbox_inches='tight', dpi=100)
plt.close()
print(f"Simulation completed. Plot saved as 'obstacle_avoidance_sqrt_cbf_with_uncertainty.png'.")
print(f"Total CBF slack used: {slack_count} times")
print(f"Final position: p = {p}, Distance to goal: {np.linalg.norm(p - p_goal):.4f}")
print(max(slack_values), min(slack_values))
