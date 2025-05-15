from casadi import *
from tracks.readDataFcn import getTrack

D_S = 0.15
D_N = 0.075
d = D_N*D_S

def dist2obs(x_e, x_o, safe, degree=2):
    s_rot = ((x_e[0] - x_o[0])*cos(x_o[2]) + (x_e[1] - x_o[1])*sin(x_o[2])) * D_N
    n_rot = (-(x_e[0] - x_o[0])*sin(x_o[2]) + (x_e[1] - x_o[1])*cos(x_o[2])) * D_S
    dist = (s_rot**degree + n_rot**degree)**(1/degree) - safe
    return dist

def FrenetBicycle(dt, degree=4, num_o=2, mode="MPC", track="LMS_Track.txt"):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()


    # load track parameters
    [s0, x_ref, y_ref, psi_ref, kapparef] = getTrack(track)
    # print(s0[5], x_ref[5], y_ref[5], psi_ref[5], kapparef[5])
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    x_ref = np.append(x_ref, x_ref[1:length])
    y_ref = np.append(y_ref, y_ref[1:length])
    psi_ref = np.append(psi_ref, psi_ref[1:length])

    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)
    x_ref = np.append(x_ref[length - 80 : length - 1], x_ref)
    y_ref = np.append(y_ref[length - 80 : length - 1], y_ref)
    psi_ref = np.append(psi_ref[length - 80 : length - 1], psi_ref)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    x_ref_s = interpolant("x_ref_s", "bspline", [s0], x_ref)
    y_ref_s = interpolant("y_ref_s", "bspline", [s0], y_ref)
    psi_ref_s = interpolant("psi_ref_s", "bspline", [s0], psi_ref)

    ## Race car parameters
    m = 0.043
    C1 = 0.5
    C2 = 15.5
    Cm1 = 0.28
    Cm2 = 0.05
    Cr0 = 0.011
    Cr2 = 0.006

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    pedal = MX.sym("pedal")
    delta = MX.sym("delta")
    x = vertcat(s, n, alpha, v, pedal, delta)

    # controls
    dpedal = MX.sym("dpedal")
    derDelta = MX.sym("derDelta")
    u = vertcat(dpedal, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # algebraic variables
    z = []



    # dynamics
    Fxd = (Cm1 - Cm2 * v) * pedal - Cr2 * v * v - Cr0 * tanh(5 * v)
    sdot = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    ndot = v * sin(alpha + C1 * delta)
    alphadot = v * C2 * delta - kapparef_s(s) * sdot
    vdot = Fxd / m * cos(C1 * delta)
    f_expl = vertcat(
        sdot,
        ndot,
        alphadot,
        vdot,
        dpedal,
        derDelta,
    )

    # constraint on forces
    a_long = Fxd / m
    a_lat = C2 * v * v * delta + a_long * sin(C1 * delta)


    # Model bounds
    model.n_min = -0.12  # width of the track [m]
    model.n_max = 0.12  # width of the track [m]
    model.s_min = 0.0
    model.s_max = 100.0/3.6
    # state bounds
    model.pedal_min = -1.0
    model.pedal_max = 1.0

    model.delta_min = -0.40  # minimum steering angle [rad]
    model.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    model.dpedal_min = -10  # -10.0  # minimum throttle change rate
    model.dpedal_max = 10  # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum lateral force [m/s^2]
    constraint.along_max = 4  # maximum lateral force [m/s^2]

    
    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength

    # Obstacle positions (parameterized)
    if num_o > 0:
        s_o = [MX.sym(f"s_o{i}") for i in range(1, num_o+1)]
        n_o = [MX.sym(f"n_o{i}") for i in range(1, num_o+1)]
        alpha_o = [MX.sym(f"alpha_o{i}") for i in range(1, num_o+1)]
        v_o = [MX.sym(f"v_o{i}") for i in range(1, num_o+1)]
        gammas = [MX.sym(f"gamma{i}") for i in range(1, num_o+1)]
        h_0 = [MX.sym(f"h_0{i}") for i in range(1, num_o+1)]
        max_h_n = MX.sym("max_h")
        x_o = [vertcat(s_o[i], n_o[i], alpha_o[i], v_o[i]) for i in range(num_o)]
        # parameters
        p = vertcat(
            *[vertcat(s_o[i], n_o[i], alpha_o[i], v_o[i], h_0[i], gammas[i]) for i in range(num_o)],  # Group [s, d, theta, dhdt] for each obstacle
            max_h_n # Append all gamma values
        )
        print(p)
        D   = [(1/gammas[i]) * (max_h_n - h_0[i]) + d for i in range(num_o)]
        # D_s = [pow((max_h_n - h_c) * sdot[i] + D_S**degree, 1/degree) for i in range(num_o)]
        # D_n = [pow((max_h_n - h_c) * ndot[i] + D_N**degree, 1/degree) for i in range(num_o)]
        
        # Barrier function for obstacle avoidance
        x_next = vertcat(s + sdot*dt, n + ndot*dt)
        h_c  = [dist2obs(x,      x_o[i], d, degree=4) for i in range(num_o)]
        h_n  = [dist2obs(x_next, x_o[i], d, degree=4) for i in range(num_o)]
        hdot = [h_n[i] - h_c[i] for i in range(num_o)]
        cbf  = [hdot[i] + gammas[i]*h_c[i] for i in range(num_o)]

        # CBF constraints for obstacle avoidance
        constraint.cbf_min = [0.0]*num_o
        constraint.cbf_max = [1e9]*num_o
        constraint.expr = vertcat(a_long, a_lat, n, pedal, delta, *cbf)

    else:
        p = []
        constraint.expr = vertcat(a_long, a_lat, n, pedal, delta)
        

    # Define model struct
    params = types.SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.params = params

    # Interpolant functions for x, y, psi
    model.x_ref_s = x_ref_s
    model.y_ref_s = y_ref_s
    model.psi_ref_s = psi_ref_s

    return model, constraint
