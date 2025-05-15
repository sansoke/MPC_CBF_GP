import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from acados_settings_dev import *
from plotFcn import *
from tracks.readDataFcn import getTrack
from time2spatial import transformProj2Orig

def setup_track_plot(ref_track):
    [Sref, Xref, Yref, Psiref, _] = getTrack(ref_track)
    lane_width = 0.24  # Lane width, modify as needed

    # Setup plot
    fig, ax = plt.subplots()
    ax.set_ylim(bottom=-1.75 * 5, top=0.35 * 5)
    ax.set_xlim(left=-1.1 * 5, right=1.6 * 5)
    ax.set_ylabel('y[m]')
    ax.set_xlabel('x[m]')

    return fig, ax, Sref, Xref, Yref, Psiref, lane_width

def draw_track(ax, Xref, Yref, Psiref, lane_width):
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

def setup_vehicle(ax, color, Carlength, Carwidth):
    vehicle_rect = plt.Rectangle((0, 0), Carlength, Carwidth, fc=color, ec='black')
    ax.add_patch(vehicle_rect)
    return vehicle_rect

def initialize_data_structs(Nsim, nx, nu, N):
    simX = np.zeros((Nsim, nx))
    simU = np.zeros((Nsim, nu))
    pred = np.ones((N, nx))*1e6
    return simX, simU, pred

def update_reference(acados_solver, s0, sref_N, N):
    sref = s0 + sref_N
    for j in range(N):
        yref = np.array([s0 + (sref - s0) * j / N, 0, 0, 0, 0, 0])
        acados_solver.set(j, "yref", yref)
    yref_N = np.array([sref, 0, 0, 0])
    acados_solver.set(N, "yref", yref_N)

def solve_ocp(acados_solver, N):
    status = acados_solver.solve()
    return status

def get_solution(acados_solver, N, nx):
    pred = np.zeros((N, nx))
    for step in range(N + 1):
        if step == 0:
            x0 = acados_solver.get(step, field_="x")
        else:
            pred[step - 1] = acados_solver.get(step, field_="x")
    return x0, pred

def update_solver_constraints(acados_solver, pred_obs1, pred_obs2, N):
    for step in range(N):
        EgosObs = np.hstack([pred_obs1[step, :2], pred_obs2[step, :2]])
        acados_solver.set(step, "p", EgosObs)

def run_simulation(track_files, model_names, colors, init_positions, lane_width, sref_N=5, T=10.00, Tf=1.0, N=50):
    fig, ax, Sref, Xref, Yref, Psiref, lane_width = setup_track_plot(track_files[0])
    draw_track(ax, Xref, Yref, Psiref, lane_width)

    models = []
    constraints = []
    solvers = []
    rects = []
    preds = []
    simXs = []
    simUs = []

    for i, (track_file, model_name, color) in enumerate(zip(track_files, model_names, colors)):
        constraint, model, acados_solver, ocp = acados_settings(Tf, N, track_file, model_name=model_name)
        constraints.append(constraint)
        models.append(model)
        solvers.append(acados_solver)
        rect = setup_vehicle(ax, color, 0.15, 0.075)
        rects.append(rect)
        simX, simU, pred = initialize_data_structs(int(T * N / Tf), model.x.rows(), model.u.rows(), N)
        preds.append(pred)
        simXs.append(simX)
        simUs.append(simU)

    tcomp_sum = 0
    tcomp_max = 0

    for i in range(int(T * N / Tf)):
        for j, (solver, model, pred) in enumerate(zip(solvers, models, preds)):
            if j == 0:
                update_reference(solver, model.x0[0], sref_N, N)
            else:
                update_solver_constraints(solver, preds[0], preds[1], N)
                update_reference(solver, model.x0[0], sref_N, N)

        # Solve OCP for all vehicles
        t = time.time()
        for solver in solvers:
            status = solve_ocp(solver, N)
            if status != 0:
                print(f"acados returned status {status} in closed loop iteration {i}.")

        elapsed = time.time() - t
        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # Get solutions for all vehicles
        for j, (solver, pred, model, simX, simU) in enumerate(zip(solvers, preds, models, simXs, simUs)):
            x0, pred = get_solution(solver, N, model.x.rows())
            solver.set(0, "lbx", pred[0])
            solver.set(0, "ubx", pred[0])
            model.x0 = pred[0]
            simX[i, :] = x0
            simU[i, :] = solver.get(0, "u")

        # Transform projections for plotting
        for j, (pred, rect, track_file) in enumerate(zip(preds, rects, track_files)):
            vehicle_xy = transformProj2Orig(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], track_file)
            update_vehicle_plot(rect, vehicle_xy[0, 0], vehicle_xy[1, 0], vehicle_xy[2, 0], 0.15, 0.075)

        plt.draw()
        plt.pause(0.001)  # Add a small pause to update the plot in real time

    # Print some stats
    print("Average computation time: {}".format(tcomp_sum / (T * N / Tf)))
    print("Maximum computation time: {}".format(tcomp_max))
    print("Average speed:{}m/s".format(np.average(simXs[0][:, 3])))
    print("Lap time: {}s".format(Tf * int(T * N / Tf) / N))

    plt.ioff()
    plt.show()
