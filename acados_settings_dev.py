#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from FrenetFrame import FrenetBicycle
from bicycle_model import bicycle_model
import scipy.linalg
import numpy as np


def acados_settings(Tf, N, track_file, num_o=2, model_name="EgoModel"):
    # create render arguments
    ocp = AcadosOcp()
    dt = Tf / N
    # export model
    model, constraint = FrenetBicycle(dt, track=track_file)
    model.name = model_name
    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    nh = constraint.expr.shape[0]

    # discretization
    ocp.dims.N = N

    # set cost
    Q = np.diag([ 4e-4, 1e-2, 6e-3, 1e-8, 1e-4, 5e-4])

    R = np.eye(nu)
    R[0, 0] = 1e-5
    R[1, 1] = 5e-5

    Qe = np.diag([ 4e-4, 1e-2, 6e-3, 1e-8, 1e-4, 5e-4])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[6, 0] = 1.0
    Vu[7, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # set intial references
    ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0])

    # setting constraints
    ocp.constraints.lbx = np.array([-12])
    ocp.constraints.ubx = np.array([12])
    ocp.constraints.idxbx = np.array([1])
    ocp.constraints.lbu = np.array([model.dpedal_min, model.ddelta_min])
    ocp.constraints.ubu = np.array([model.dpedal_max, model.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])
    if num_o > 0:
        ocp.constraints.lh = np.concatenate((np.array([constraint.along_min, constraint.alat_min, model.n_min, model.pedal_min, model.delta_min]), np.array(constraint.cbf_min)))
        ocp.constraints.uh = np.concatenate((np.array([constraint.along_max, constraint.alat_max, model.n_max, model.pedal_max, model.delta_max]), np.array(constraint.cbf_max)))
        # print(ocp.constraints.lh)
        # print(ocp.constraints.uh)
    else:
        ocp.constraints.lh = np.array(
            [
                model.along_min,
                model.alat_min,
                model.n_min,
                model.pedal_min,
                model.delta_min,
            ]
        )
        ocp.constraints.uh = np.array(
            [
                model.along_max,
                model.alat_max,
                model.n_max,
                model.pedal_max,
                model.delta_max,
            ]
        )
    # Set parameter values
    # ocp.parameter_values = np.zeros(model.p.shape[0])
    ocp.parameter_values = np.array([1e4, 1e4, 0.0, 0.0, 0.0, 1.0]*num_o + [0.0])

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver, ocp

def acados_setting(Tf, N, track_file, model_name="EgoModel"):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint = bicycle_model(track_file)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # set dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N
    ns = 1
    nsh = 1

    # set cost
    Q = np.diag([ 4e-4, 1e-2, 6e-3, 1e-8, 1e-4, 5e-4])

    R = np.eye(nu)
    R[0, 0] = 1e-5
    R[1, 1] = 5e-5

    Qe = np.diag([ 4e-4, 1e-2, 6e-3, 1e-8, 1e-4, 5e-4])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[6, 0] = 1.0
    Vu[7, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.Zl = 0 * np.ones((ns,))
    ocp.cost.zu = 100 * np.ones((ns,))
    ocp.cost.Zu = 0 * np.ones((ns,))

    # set intial references
    ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0])

    # setting constraints
    ocp.constraints.lbx = np.array([-12])
    ocp.constraints.ubx = np.array([12])
    ocp.constraints.idxbx = np.array([1])
    ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min])
    ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])
    # ocp.constraints.lsbx=np.zero s([1])
    # ocp.constraints.usbx=np.zeros([1])
    # ocp.constraints.idxsbx=np.array([1])
    ocp.constraints.lh = np.array(
        [
            constraint.along_min,
            constraint.alat_min,
            model.n_min,
            model.throttle_min,
            model.delta_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            constraint.along_max,
            constraint.alat_max,
            model.n_max,
            model.throttle_max,
            model.delta_max,
        ]
    )
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array([2])

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3

    # ocp.solver_options.qp_solver_tol_stat = 1e-2
    # ocp.solver_options.qp_solver_tol_eq = 1e-2
    # ocp.solver_options.qp_solver_tol_ineq = 1e-2
    # ocp.solver_options.qp_solver_tol_comp = 1e-2

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver, ocp