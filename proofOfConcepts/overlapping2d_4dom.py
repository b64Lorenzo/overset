#!/usr/bin/env python3
"""
2D hierarchical four-subdomain validation

           D3 | D4
          ---------
           D1 | D2

The four subdomains are solved independently using perturbed
interface data. 
"""

from src.utils.logger import Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.interpolate import (
    LinearNDInterpolator,
    interp1d,
)

from skfem import (
    BilinearForm,
    LinearForm,
)

from skfem.helpers import (
    dot,
    grad,
)

from src.solver.fem2d import (
    solve_rect,
)

from src.reconstruction.overlap2d import (
    Domain2D,
    DomainSolution2D,
    SolutionDomain2D,
    OversetInterfaceCorrector2D,
    ObservationStrategy,
)

from src.problems.elliptic2d import (
    Elliptic2DProblem,
    Elliptic2DCoefficients,
)


from src.plotlib.plot import (
    plot_field,
    plot_surface,
    plot_subdomains,
    export_subdomains,
    build_global_interpolator,
)

from src.plotlib.overlap4d_plots import (
    plot_overlap4d_subdomains,
    plot_reference_solution,
    plot_reconstructed_solution,
    plot_error_before,
    plot_error_after,
    postprocess_overlap4d,
)

# ==========================================================
# PLOTTING
# ==========================================================

plt.rcParams.update(
    {
        "font.size": 14
    }
)

# ==========================================================
# LOGGER
# ==========================================================

log_mgr = Logger(
    name="overlap2d_4dom",
    run_prefix="overlap2d_4dom",
)

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()


# ==========================================================
# CASE BUILDER
# ==========================================================

def build_case(
    n_modes=5,
    n_obs=10,
    overlap_x_left=1,
    overlap_x_right=1,
    overlap_y_top=1,
    overlap_y_bottom=1,
):

    # ------------------------------------------------------
    # GEOMETRY
    # ------------------------------------------------------

    Lx = 100.0
    Ly = 100.0

    xI = 0.5 * Lx
    yI = 0.5 * Ly

    nx_full = 300
    ny_full = 300

    nx_sub = nx_full // 2
    ny_sub = ny_full // 2

    hx_left  = xI / nx_sub
    hx_right = (Lx - xI) / nx_sub

    hy_south = yI / ny_sub
    hy_north = (Ly - yI) / ny_sub

    # ------------------------------------------------------
    # OVERSET INTERFACES
    #
    # xW < xI < xE
    # yS < yI < yN
    # ------------------------------------------------------

    xW = (
        xI
        - overlap_x_right * hx_right
    )

    xE = (
        xI
        + overlap_x_left * hx_left
    )

    yS = (
        yI
        - overlap_y_top * hy_north
    )

    yN = (
        yI
        + overlap_y_bottom * hy_south
    )

    # ------------------------------------------------------
    # WEIGHTS
    # ------------------------------------------------------

    def w1(x, y):

        wx = (xE - x) / (xE - xW)
        wy = (yN - y) / (yN - yS)

        return max(0.0, wx) * max(0.0, wy)


    def w2(x, y):

        wx = (x - xW) / (xE - xW)
        wy = (yN - y) / (yN - yS)

        return max(0.0, wx) * max(0.0, wy)


    def w3(x, y):

        wx = (xE - x) / (xE - xW)
        wy = (y - yS) / (yN - yS)

        return max(0.0, wx) * max(0.0, wy)


    def w4(x, y):

        wx = (x - xW) / (xE - xW)
        wy = (y - yS) / (yN - yS)

        return max(0.0, wx) * max(0.0, wy)

    # ------------------------------------------------------
    # BOUNDARY COORDINATES
    # ------------------------------------------------------

    yD12 = np.linspace(
        0.0,
        yN,
        ny_sub + 1,
    )

    yD34 = np.linspace(
        yS,
        Ly,
        ny_sub + 1,
    )

    xD13 = np.linspace(
        0.0,
        xE,
        nx_sub + 1,
    )

    xD24 = np.linspace(
        xW,
        Lx,
        nx_sub + 1,
    )

    # ------------------------------------------------------
    # PDE
    # ------------------------------------------------------

    problem = Elliptic2DProblem(

        Elliptic2DCoefficients(

            k=25.0,

            h0=30.0,

            p0=10.0,
            p_amp=64.0,

            kxP=0.64,
            kyP=0.25,

            A_f=2.0,

            fx_wave=0.25,
            fy_wave=0.15,
        )
    )

    diffusion_form,reaction_form,rhs_form = (
        problem.forms()
    )

    # ------------------------------------------------------
    # BC
    # ------------------------------------------------------

    def bc_west(y): return 0.0
    def bc_east(y): return 0.0
    def bc_south(x): return 0.0
    def bc_north(x): return 0.0

    full_bc = {

        "W": bc_west,
        "E": bc_east,
        "S": bc_south,
        "N": bc_north,
    }

    # ------------------------------------------------------
    # REFERENCE
    # ------------------------------------------------------

    # ==========================================================
    # REFERENCE SOLUTION
    # ==========================================================

    logger.info(
        "Computing reference solution"
    )

    full_bc = {

        "W": bc_west,

        "E": bc_east,

        "S": bc_south,

        "N": bc_north,
    }

    meshF, basisF, uF = solve_rect(
        xmin=0.0,
        xmax=Lx,
        ymin=0.0,
        ymax=Ly,
        nx=nx_full,
        ny=ny_full,
        diffusion_form=diffusion_form,
        reaction_form=reaction_form,
        rhs_form=rhs_form,
        boundary_conditions=full_bc,
    )

    interp_full = LinearNDInterpolator(
        np.c_[
            basisF.doflocs[0],
            basisF.doflocs[1],
        ],
        uF,
    )
    prev_noise = 0.00064

    def interp_prev_D1(x, y):
        return (
            interp_full(x, y)
            + prev_noise
            * np.sin(2.0*np.pi*x/Lx)
            * np.sin(np.pi*y/Ly)
        )

    def interp_prev_D2(x, y):
        return (
            interp_full(x, y)
            - prev_noise
            * np.sin(2.0*np.pi*x/Lx)
            * np.sin(np.pi*y/Ly)
        )

    def interp_prev_D3(x, y):
        return (
            interp_full(x, y)
            + prev_noise
            * np.cos(np.pi*x/Lx)
            * np.sin(2.0*np.pi*y/Ly)
        )

    def interp_prev_D4(x, y):
        return (
            interp_full(x, y)
            - prev_noise
            * np.cos(np.pi*x/Lx)
            * np.sin(2.0*np.pi*y/Ly)
        )
    
    # ==========================================================
    # EXACT TRACES ON OVERSET BOUNDARIES
    # ==========================================================

    # E/W traces

    u12_E_exact = interp_full(
        np.full_like(yD12, xE),
        yD12,
    )

    u12_W_exact = interp_full(
        np.full_like(yD12, xW),
        yD12,
    )

    u34_E_exact = interp_full(
        np.full_like(yD34, xE),
        yD34,
    )

    u34_W_exact = interp_full(
        np.full_like(yD34, xW),
        yD34,
    )

    # N/S traces

    u1_N_exact = interp_full(
        xD13,
        np.full_like(xD13, yN),
    )

    u2_N_exact = interp_full(
        xD24,
        np.full_like(xD24, yN),
    )

    u3_S_exact = interp_full(
        xD13,
        np.full_like(xD13, yS),
    )

    u4_S_exact = interp_full(
        xD24,
        np.full_like(xD24, yS),
    )


    eps = 30.0

    u12_E = (
        u12_E_exact
        + eps*np.sin(2*
            np.pi*yD12/yN
        )
    )

    u12_W = (
        u12_W_exact
        - eps*np.sin(2*
            np.pi*yD12/yN
        )
    )

    u34_E = (
        u34_E_exact
        + eps*np.sin(2*
            np.pi*(yD34-yS)/(Ly-yS)
        )
    )

    u34_W = (
        u34_W_exact
        - eps*np.sin(2*
            np.pi*(yD34-yS)/(Ly-yS)
        )
    )


    u1_N = (
        u1_N_exact
        + eps*np.sin(2*
            np.pi*xD13/xE
        )
    )

    u2_N = (
        u2_N_exact
        - eps*np.sin(2*
            np.pi*(xD24-xW)/(Lx-xW)
        )
    )

    u3_S = (
        u3_S_exact
        + eps*np.sin(2*
            np.pi*xD13/xE
        )
    )

    u4_S = (
        u4_S_exact
        - eps*np.sin(2*
            np.pi*(xD24-xW)/(Lx-xW)
        )
    )


    # ==========================================================
    # DOMAIN D1
    # ==========================================================

    logger.info("Solving D1")

    bc1 = {
        "W": bc_west,
        "S": bc_south,

        "E": {
            "coords": yD12,
            "values": u12_E,
        },

        "N": {
            "coords": xD13,
            "values": u1_N,
        },
    }
    mesh1, basis1, u1 = solve_rect(

        xmin=0.0,
        xmax=xE,

        ymin=0.0,
        ymax=yN,

        nx=nx_sub,
        ny=ny_sub,

        diffusion_form=diffusion_form,
        reaction_form=reaction_form,
        rhs_form=rhs_form,

        boundary_conditions=bc1,
    )

    # ==========================================================
    # DOMAIN D2
    # ==========================================================

    logger.info("Solving D2")

    bc2 = {
        "W": {
            "coords": yD12,
            "values": u12_W,
        },

        "E": bc_east,
        "S": bc_south,

        "N": {
            "coords": xD24,
            "values": u2_N,
        },
    }
    mesh2, basis2, u2 = solve_rect(

        xmin=xW,
        xmax=Lx,

        ymin=0.0,
        ymax=yN,

        nx=nx_sub,
        ny=ny_sub,

        diffusion_form=diffusion_form,
        reaction_form=reaction_form,
        rhs_form=rhs_form,

        boundary_conditions=bc2,
    )

    # ==========================================================
    # DOMAIN D3
    # ==========================================================

    logger.info("Solving D3")

    bc3 = {
        "W": bc_west,

        "E": {
            "coords": yD34,
            "values": u34_E,
        },

        "S": {
            "coords": xD13,
            "values": u3_S,
        },

        "N": bc_north,
    }

    mesh3, basis3, u3 = solve_rect(

        xmin=0.0,
        xmax=xE,

        ymin=yS,
        ymax=Ly,
        nx=nx_sub,
        ny=ny_sub,

        diffusion_form=diffusion_form,
        reaction_form=reaction_form,
        rhs_form=rhs_form,

        boundary_conditions=bc3,
    )
    # ==========================================================
    # DOMAIN D4
    # ==========================================================

    logger.info("Solving D4")

    bc4 = {
        "W": {
            "coords": yD34,
            "values": u34_W,
        },

        "E": bc_east,

        "S": {
            "coords": xD24,
            "values": u4_S,
        },

        "N": bc_north,
    }

    mesh4, basis4, u4 = solve_rect(

        xmin=xW,
        xmax=Lx,

        ymin=yS,
        ymax=Ly,

        nx=nx_sub,
        ny=ny_sub,

        diffusion_form=diffusion_form,
        reaction_form=reaction_form,
        rhs_form=rhs_form,

        boundary_conditions=bc4,
    )

    # ==========================================================
    # DOMAIN INTERPOLATORS
    # ==========================================================

    interp_D1 = LinearNDInterpolator(
        np.c_[basis1.doflocs[0], basis1.doflocs[1]],
        u1,
    )

    interp_D2 = LinearNDInterpolator(
        np.c_[basis2.doflocs[0], basis2.doflocs[1]],
        u2,
    )

    interp_D3 = LinearNDInterpolator(
        np.c_[basis3.doflocs[0], basis3.doflocs[1]],
        u3,
    )

    interp_D4 = LinearNDInterpolator(
        np.c_[basis4.doflocs[0], basis4.doflocs[1]],
        u4,
    )

    domain1 = Domain2D(
        a=0.0,
        b=xE,
        c=0.0,
        d=yN,
        F0=problem.h(0.0,0.0),
        k=problem.k,
        interfaces=("E", "N"),
        n_modes=n_modes,
        weight_function=w1
    )

    domain2 = Domain2D(
        a=xW,
        b=Lx,
        c=0.0,
        d=yN,
        F0=problem.h(Lx, 0.0),
        k=problem.k,
        interfaces=("W", "N"),
        n_modes=n_modes,
        weight_function=w2
    )

    domain3 = Domain2D(
        a=0.0,
        b=xE,
        c=yS,
        d=Ly,
        F0=problem.h(0.0, Ly),
        k=problem.k,
        interfaces=("E", "S"),
        n_modes=n_modes,
        weight_function=w3
    )

    domain4 = Domain2D(
        a=xW,
        b=Lx,
        c=yS,
        d=Ly,
        F0=problem.h(Lx, Ly),
        k=problem.k,
        interfaces=("W", "S"),
        n_modes=n_modes,
        weight_function=w4
    )

    D1 = DomainSolution2D(
        domain=domain1,
        solution=u1,
        x=basis1.doflocs[0],
        y=basis1.doflocs[1],
        interpolator=interp_D1,
    )

    D2 = DomainSolution2D(
        domain=domain2,
        solution=u2,
        x=basis2.doflocs[0],
        y=basis2.doflocs[1],
        interpolator=interp_D2,
    )

    D3 = DomainSolution2D(
        domain=domain3,
        solution=u3,
        x=basis3.doflocs[0],
        y=basis3.doflocs[1],
        interpolator=interp_D3,
    )

    D4 = DomainSolution2D(
        domain=domain4,
        solution=u4,
        x=basis4.doflocs[0],
        y=basis4.doflocs[1],
        interpolator=interp_D4,
    )

    U = SolutionDomain2D(
        domains=(
            D1,
            D2,
            D3,
            D4,
        )
    )


    return {
        "problem": problem,

        "Lx": Lx,
        "Ly": Ly,

        "basisF": basisF,
        "uF": uF,

        "interp_full": interp_full,

        "U": U,

        "D1": D1,
        "D2": D2,
        "D3": D3,
        "D4": D4,

        "interp_D1": interp_D1,
        "interp_D2": interp_D2,
        "interp_D3": interp_D3,
        "interp_D4": interp_D4,

        "basis1": basis1,
        "basis2": basis2,
        "basis3": basis3,
        "basis4": basis4,

        "u1": u1,
        "u2": u2,
        "u3": u3,
        "u4": u4,

        "xW": xW,
        "xE": xE,

        "yS": yS,
        "yN": yN,

        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
    }


# ==========================================================
# FPS OBSERVATIONS
# ==========================================================

def build_fps_observations(
    xI_left,
    xI_right,
    yI_top,
    yI_bottom,
    n_obs,
):

    xx = np.linspace(
        xI_right,
        xI_left,
        100,
    )

    yy = np.linspace(
        yI_top,
        yI_bottom,
        100,
    )

    XX, YY = np.meshgrid(
        xx,
        yy,
    )

    candidates = np.c_[
        XX.ravel(),
        YY.ravel(),
    ]

    return ObservationStrategy.fps(
        candidates,
        n_obs,
    )

def single_run():
    logger.info("=" * 80)
    logger.info("Running single run")
    n_obs_line = 2
    case = build_case(
        n_modes=2,
        n_obs=n_obs_line,
        overlap_x_left=1,
        overlap_x_right=1,
        overlap_y_top=1,
        overlap_y_bottom=1,
    )

    U = case["U"]

    basisF = case["basisF"]
    uF = case["uF"]

    xW  = case["xW"]
    xE = case["xE"]

    yS    = case["yS"]
    yN = case["yN"]

    u1 = case["u1"]
    u2 = case["u2"]
    u3 = case["u3"]
    u4 = case["u4"]

    basis1 = case["basis1"]
    basis2 = case["basis2"]
    basis3 = case["basis3"]
    basis4 = case["basis4"]

    Lx = case["Lx"]
    Ly = case["Ly"]
    
        # ==========================================================
    # OBSERVATION POINTS
    # ==========================================================
    #
    # D1-D2 :
    # y in [0,yN]
    #
    # D3-D4 :
    # y in [yS,Ly]
    #
    # E/W reconstruction only
    # ==========================================================

    n_obs_line = 4

    # ----------------------------------------------------------
    # D1-D2 observation lines
    # ----------------------------------------------------------

    yD12 = np.linspace(
        0.0,
        yN,
        n_obs_line,
    )

    obs_D12_W = np.column_stack(
        [
            np.full(
                n_obs_line,
                xW,
            ),
            yD12,
        ]
    )

    obs_D12_E = np.column_stack(
        [
            np.full(
                n_obs_line,
                xE,
            ),
            yD12,
        ]
    )




    # ----------------------------------------------------------
    # D3-D4 observation lines
    # ----------------------------------------------------------

    yD34 = np.linspace(
        yS,
        Ly,
        n_obs_line,
    )

    obs_D34_W = np.column_stack(
        [
            np.full(
                n_obs_line,
                xW,
            ),
            yD34,
        ]
    )

    obs_D34_E = np.column_stack(
        [
            np.full(
                n_obs_line,
                xE,
            ),
            yD34,
        ]
    )

    xD13_obs = np.linspace(
        0.0,
        xE,
        n_obs_line,
    )

    xD24_obs = np.linspace(
        xW,
        Lx,
        n_obs_line,
    )

    obs_D13_N = np.column_stack(
        [
            xD13_obs,
            np.full(
                n_obs_line,
                yN,
            ),
        ]
    )

    obs_D24_N = np.column_stack(
        [
            xD24_obs,
            np.full(
                n_obs_line,
                yN,
            ),
        ]
    )

    obs_D13_S = np.column_stack(
        [
            xD13_obs,
            np.full(
                n_obs_line,
                yS,
            ),
        ]
    )

    obs_D24_S = np.column_stack(
        [
            xD24_obs,
            np.full(
                n_obs_line,
                yS,
            ),
        ]
    )

    # ----------------------------------------------------------
    # GLOBAL CONSTRAINT SET
    # ----------------------------------------------------------

    constraint_points = np.vstack(
        [
            obs_D12_W,
            obs_D12_E,

            obs_D34_W,
            obs_D34_E,

            obs_D13_N,
            obs_D24_N,

            obs_D13_S,
            obs_D24_S,
        ]
    )


    # ==========================================================
    # GLOBAL CORRECTION
    # ==========================================================

    logger.info(
        f"Unknown coefficients = "
        f"{U.total_unknowns}"
    )

    logger.info(
        f"Constraint points = "
        f"{len(constraint_points)}"
    )

    result = (
        OversetInterfaceCorrector2D
        .correct(
            U=U,
            constraint_points=constraint_points,
        )
    )

    # ==========================================================
    # CORRECTION OUTPUTS
    # ==========================================================

    u1f = result.corrected[0]
    u2f = result.corrected[1]
    u3f = result.corrected[2]
    u4f = result.corrected[3]

    corr1 = result.corrections[0]
    corr2 = result.corrections[1]
    corr3 = result.corrections[2]
    corr4 = result.corrections[3]

    coeffs = result.coefficients

    logger.info(
        f"||coeff|| = "
        f"{np.linalg.norm(coeffs):.6e}"
    )

    logger.info(
        f"n_coeff = "
        f"{len(coeffs)}"
    )

    # ==========================================================
    # SUBDOMAIN DATA
    # ==========================================================

    subdomains = [

        {
            "name": "D1",
            "basis": basis1,
            "u": u1,
            "uc": u1f,
            "correction": corr1,
        },

        {
            "name": "D2",
            "basis": basis2,
            "u": u2,
            "uc": u2f,
            "correction": corr2,
        },

        {
            "name": "D3",
            "basis": basis3,
            "u": u3,
            "uc": u3f,
            "correction": corr3,
        },

        {
            "name": "D4",
            "basis": basis4,
            "u": u4,
            "uc": u4f,
            "correction": corr4,
        },

    ]

    # ==========================================================
    # ERROR ANALYSIS
    # ==========================================================

    analysis = postprocess_overlap4d(
        basisF,
        uF,
        subdomains,
        result,
    )

    Xf = analysis["X"]
    Yf = analysis["Y"]

    u_rec = analysis["u_rec"]

    err_before = analysis["err_before"]
    err_after = analysis["err_after"]

    L2_before = analysis["L2_before"]
    L2_after = analysis["L2_after"]

    logger.info(
        f"L2 before = {L2_before:.6e}"
    )

    logger.info(
        f"L2 after = {L2_after:.6e}"
    )

    logger.info(
        f"Improvement = "
        f"{L2_before/L2_after:.3f}"
    )

    

    plot_error_before(
        Xf,
        Yf,
        err_before,
        run_dir,
    )

    plot_error_after(
        Xf,
        Yf,
        err_after,
        run_dir,
    )


# ==========================================================
# SINGLE FPS RUN
# ==========================================================

def single_fps_run():

    logger.info("=" * 80)
    logger.info("Running single FPS reconstruction")

    n_modes = 2

    # 8 interfaces × n_modes unknowns
    n_obs = 16

    case = build_case(

        n_modes=n_modes,

        n_obs=n_obs,

        overlap_x_left=8,
        overlap_x_right=8,

        overlap_y_top=8,
        overlap_y_bottom=8,
    )

    U = case["U"]

    basisF = case["basisF"]
    uF = case["uF"]

    Xf = basisF.doflocs[0]
    Yf = basisF.doflocs[1]

    xW = case["xW"]
    xE = case["xE"]

    yS = case["yS"]
    yN = case["yN"]

    # ======================================================
    # FPS OBSERVATIONS INSIDE OVERLAP REGION
    # ======================================================

    xx = np.linspace(
        xW,
        xE,
        100,
    )

    yy = np.linspace(
        yS,
        yN,
        100,
    )

    XX, YY = np.meshgrid(
        xx,
        yy,
    )

    candidate_points = np.c_[
        XX.ravel(),
        YY.ravel(),
    ]

    constraint_points = (
        ObservationStrategy.fps(
            candidate_points,
            n_obs,
        )
    )

    logger.info(
        f"Unknown coefficients = "
        f"{U.total_unknowns}"
    )

    logger.info(
        f"Constraint points = "
        f"{len(constraint_points)}"
    )

    result = (
        OversetInterfaceCorrector2D
        .correct(
            U=U,
            constraint_points=
            constraint_points,
        )
    )

    coeffs = result.coefficients

    logger.info(
        f"||coeff|| = "
        f"{np.linalg.norm(coeffs):.6e}"
    )

    logger.info(
        f"n_coeff = "
        f"{len(coeffs)}"
    )

    logger.info(
        f"Condition = "
        f"{result.condition_number:.6e}"
    )

    logger.info(
        f"Residual = "
        f"{result.residual_norm:.6e}"
    )

    # ======================================================
    # SUBDOMAIN DATA
    # ======================================================

    subdomains = [

        {
            "name": "D1",
            "basis": case["basis1"],
            "u": case["u1"],
            "uc": result.corrected[0],
            "correction":
            result.corrections[0],
        },

        {
            "name": "D2",
            "basis": case["basis2"],
            "u": case["u2"],
            "uc": result.corrected[1],
            "correction":
            result.corrections[1],
        },

        {
            "name": "D3",
            "basis": case["basis3"],
            "u": case["u3"],
            "uc": result.corrected[2],
            "correction":
            result.corrections[2],
        },

        {
            "name": "D4",
            "basis": case["basis4"],
            "u": case["u4"],
            "uc": result.corrected[3],
            "correction":
            result.corrections[3],
        },
    ]

    # ======================================================
    # ERROR ANALYSIS
    # ======================================================

    analysis = postprocess_overlap4d(
        basisF,
        uF,
        subdomains,
        result,
    )

    logger.info(
        f"L2 before = "
        f"{analysis['L2_before']:.6e}"
    )

    logger.info(
        f"L2 after = "
        f"{analysis['L2_after']:.6e}"
    )

    logger.info(
        f"Improvement = "
        f"{analysis['L2_before']/analysis['L2_after']:.3f}"
    )

    plot_error_before(
        analysis["X"],
        analysis["Y"],
        analysis["err_before"],
        run_dir,
    )

    plot_error_after(
        analysis["X"],
        analysis["Y"],
        analysis["err_after"],
        run_dir,
    )

    return result

def run_overlap_width_comparison():

    logger.info("=" * 80)

    logger.info(
        "Running overlap-width comparison study "
        "(Two-Line vs FPS)"
    )

    overlap_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
    ]

    rows = []

    for overlap in overlap_sizes:

        logger.info(
            f"Overlap cells = {overlap}"
        )

        # ==================================================
        # CASE
        # ==================================================

        case = build_case(

            n_modes=2,

            overlap_x_left=overlap,
            overlap_x_right=overlap,

            overlap_y_top=overlap,
            overlap_y_bottom=overlap,
        )

        U = case["U"]

        xW = case["xW"]
        xE = case["xE"]

        yS = case["yS"]
        yN = case["yN"]

        Lx = case["Lx"]
        Ly = case["Ly"]

        basisF = case["basisF"]
        uF = case["uF"]

        Xf = basisF.doflocs[0]
        Yf = basisF.doflocs[1]

        width_x = xE - xW
        width_y = yN - yS

        # ==================================================
        # TWO-LINE OBSERVATIONS
        #
        # n_modes=2
        # unknowns=16
        # observations=16
        # ==================================================

        n_obs_line = 2

        yD12 = np.linspace(
            0.0,
            yN,
            n_obs_line,
        )

        yD34 = np.linspace(
            yS,
            Ly,
            n_obs_line,
        )

        xD13 = np.linspace(
            0.0,
            xE,
            n_obs_line,
        )

        xD24 = np.linspace(
            xW,
            Lx,
            n_obs_line,
        )

        obs_D12_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD12,
            ]
        )

        obs_D12_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD12,
            ]
        )

        obs_D34_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD34,
            ]
        )

        obs_D34_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD34,
            ]
        )

        obs_D13_N = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D13_S = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        obs_D24_N = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D24_S = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        constraint_lines = np.vstack(
            [
                obs_D12_W,
                obs_D12_E,

                obs_D34_W,
                obs_D34_E,

                obs_D13_N,
                obs_D13_S,

                obs_D24_N,
                obs_D24_S,
            ]
        )

        # ==================================================
        # FPS OBSERVATIONS
        # ==================================================

        xx = np.linspace(
            xW,
            xE,
            100,
        )

        yy = np.linspace(
            yS,
            yN,
            100,
        )

        XX, YY = np.meshgrid(
            xx,
            yy,
        )

        candidate_points = np.c_[
            XX.ravel(),
            YY.ravel(),
        ]

        constraint_fps = (
            ObservationStrategy.fps(
                candidate_points,
                len(constraint_lines),
            )
        )

        # ==================================================
        # TWO-LINE RECONSTRUCTION
        # ==================================================

        result_lines = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_lines,
            )
        )

        interp_lines = LinearNDInterpolator(
            np.c_[
                result_lines.full_x,
                result_lines.full_y,
            ],
            result_lines.full_solution,
            fill_value=np.nan,
        )

        u_lines = interp_lines(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_lines)

        error_lines = np.linalg.norm(
            u_lines[mask]
            -
            uF[mask]
        )

        # ==================================================
        # FPS RECONSTRUCTION
        # ==================================================

        result_fps = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_fps,
            )
        )

        interp_fps = LinearNDInterpolator(
            np.c_[
                result_fps.full_x,
                result_fps.full_y,
            ],
            result_fps.full_solution,
            fill_value=np.nan,
        )

        u_fps = interp_fps(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_fps)

        error_fps = np.linalg.norm(
            u_fps[mask]
            -
            uF[mask]
        )

        rows.append(
            {
                "overlap": overlap,

                "width_x": width_x,
                "width_y": width_y,

                "error_lines":
                error_lines,

                "error_fps":
                error_fps,

                "condition_lines":
                result_lines.condition_number,

                "condition_fps":
                result_fps.condition_number,

                "residual_lines":
                result_lines.residual_norm,

                "residual_fps":
                result_fps.residual_norm,
            }
        )

        logger.info(
            f"width={width_x:.4f} "
            f"lines={error_lines:.6e} "
            f"fps={error_fps:.6e}"
        )

    # ======================================================
    # DATAFRAME
    # ======================================================

    df = pd.DataFrame(rows)

    # ======================================================
    # RATES
    # ======================================================

    df["rate_lines"] = np.nan
    df["rate_fps"] = np.nan

    for i in range(1, len(df)):

        h0 = df.loc[i - 1, "width_x"]
        h1 = df.loc[i, "width_x"]

        e0 = df.loc[i - 1, "error_lines"]
        e1 = df.loc[i, "error_lines"]

        df.loc[i, "rate_lines"] = (
            np.log(e0 / e1)
            /
            np.log(h0 / h1)
        )

        e0 = df.loc[i - 1, "error_fps"]
        e1 = df.loc[i, "error_fps"]

        df.loc[i, "rate_fps"] = (
            np.log(e0 / e1)
            /
            np.log(h0 / h1)
        )

    # ======================================================
    # PRINT RATES
    # ======================================================

    logger.info("")
    logger.info(
        "Observed convergence rates"
    )

    for _, row in df.iterrows():

        logger.info(
            f"width={row['width_x']:.6f} "
            f"rate_lines={row['rate_lines']} "
            f"rate_fps={row['rate_fps']}"
        )

    # ======================================================
    # ERROR COMPARISON
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["width_x"],
        df["error_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.loglog(
        df["width_x"],
        df["error_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(
        True,
        which="both",
    )

    # ------------------------------------------
    # reference slopes
    # ------------------------------------------

    h = df["width_x"].to_numpy()

    h_ref = np.array([
        h.min(),
        h.max(),
    ])

    # anchor near the first data point

    c1 = df["error_lines"].iloc[0] * h[0]
    c2 = df["error_lines"].iloc[0] * h[0]**2

    err_order1 = c1 / h_ref
    err_order2 = c2 / h_ref**2

    plt.loglog(
        h_ref,
        err_order1,
        "--k",
        linewidth=2,
        label=r"$O(h^{-1})$",
    )

    plt.loglog(
        h_ref,
        err_order2,
        "--r",
        linewidth=2,
        label=r"$O(h^{-2})$",
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "overlap_width_error_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================================
    # CONDITION COMPARISON
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["width_x"],
        df["condition_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.loglog(
        df["width_x"],
        df["condition_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "overlap_width_condition_comparison.png",
        dpi=400,
    )

    plt.close()

    return df


def run_modal_comparison():

    logger.info("=" * 80)

    logger.info(
        "Running modal comparison study "
        "(Two-Line vs FPS)"
    )

    mode_counts = [
        1,
        2,
        4,
        8,
        16,
        32, 
    ]

    rows = []

    for nm in mode_counts:

        logger.info(
            f"Modes/interface = {nm}"
        )

        case = build_case(
            n_modes=nm,
            overlap_x_left=8,
            overlap_x_right=8,  
            overlap_y_top=8,
            overlap_y_bottom=8,
        )

        U = case["U"]

        basisF = case["basisF"]
        uF = case["uF"]

        Xf = basisF.doflocs[0]
        Yf = basisF.doflocs[1]

        xW = case["xW"]
        xE = case["xE"]

        yS = case["yS"]
        yN = case["yN"]

        Lx = case["Lx"]
        Ly = case["Ly"]

        # ==================================================
        # TWO-LINE OBSERVATIONS
        # ==================================================

        n_obs_line = nm

        yD12 = np.linspace(
            0.0,
            yN,
            n_obs_line,
        )

        yD34 = np.linspace(
            yS,
            Ly,
            n_obs_line,
        )

        xD13 = np.linspace(
            0.0,
            xE,
            n_obs_line,
        )

        xD24 = np.linspace(
            xW,
            Lx,
            n_obs_line,
        )

        obs_D12_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD12,
            ]
        )

        obs_D12_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD12,
            ]
        )

        obs_D34_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD34,
            ]
        )

        obs_D34_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD34,
            ]
        )

        obs_D13_N = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D13_S = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        obs_D24_N = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D24_S = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        constraint_lines = np.vstack(
            [
                obs_D12_W,
                obs_D12_E,

                obs_D34_W,
                obs_D34_E,

                obs_D13_N,
                obs_D13_S,

                obs_D24_N,
                obs_D24_S,
            ]
        )

        # ==================================================
        # FPS OBSERVATIONS
        # ==================================================

        xx = np.linspace(
            xW,
            xE,
            100,
        )

        yy = np.linspace(
            yS,
            yN,
            100,
        )

        XX, YY = np.meshgrid(
            xx,
            yy,
        )

        candidate_points = np.c_[
            XX.ravel(),
            YY.ravel(),
        ]

        constraint_fps = (
            ObservationStrategy.fps(
                candidate_points,
                len(constraint_lines),
            )
        )

        logger.info(
            f"Unknowns = "
            f"{U.total_unknowns}"
        )

        logger.info(
            f"Line obs = "
            f"{len(constraint_lines)}"
        )

        logger.info(
            f"FPS obs = "
            f"{len(constraint_fps)}"
        )

        # ==================================================
        # TWO-LINE RECONSTRUCTION
        # ==================================================

        result_lines = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_lines,
            )
        )

        interp_lines = LinearNDInterpolator(
            np.c_[
                result_lines.full_x,
                result_lines.full_y,
            ],
            result_lines.full_solution,
            fill_value=np.nan,
        )

        u_lines = interp_lines(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_lines)

        error_lines = np.linalg.norm(
            u_lines[mask]
            -
            uF[mask]
        )

        # ==================================================
        # FPS RECONSTRUCTION
        # ==================================================

        result_fps = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_fps,
            )
        )

        interp_fps = LinearNDInterpolator(
            np.c_[
                result_fps.full_x,
                result_fps.full_y,
            ],
            result_fps.full_solution,
            fill_value=np.nan,
        )

        u_fps = interp_fps(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_fps)

        error_fps = np.linalg.norm(
            u_fps[mask]
            -
            uF[mask]
        )

        rows.append(
            {
                "n_modes": nm,

                "n_unknowns":
                U.total_unknowns,

                "n_obs":
                len(constraint_lines),

                "error_lines":
                error_lines,

                "error_fps":
                error_fps,

                "condition_lines":
                result_lines.condition_number,

                "condition_fps":
                result_fps.condition_number,

                "residual_lines":
                result_lines.residual_norm,

                "residual_fps":
                result_fps.residual_norm,
            }
        )

        logger.info(
            f"Lines error = "
            f"{error_lines:.6e}"
        )

        logger.info(
            f"FPS error = "
            f"{error_fps:.6e}"
        )

    # ======================================================
    # DATAFRAME
    # ======================================================

    df = pd.DataFrame(rows)

    # ======================================================
    # EXPONENTIAL FITS
    # ======================================================

    coeff_lines = np.polyfit(
        df["n_modes"],
        np.log(df["error_lines"]),
        1,
    )

    beta_lines = -coeff_lines[0]
    C_lines = np.exp(coeff_lines[1])

    coeff_fps = np.polyfit(
        df["n_modes"],
        np.log(df["error_fps"]),
        1,
    )

    beta_fps = -coeff_fps[0]
    C_fps = np.exp(coeff_fps[1])

    Nm = df["n_modes"].to_numpy()

    fit_lines = (
        C_lines
        * np.exp(
            -beta_lines * Nm
        )
    )

    fit_fps = (
        C_fps
        * np.exp(
            -beta_fps * Nm
        )
    )

    logger.info(
        f"Two-line beta = {beta_lines:.6e}"
    )

    logger.info(
        f"FPS beta = {beta_fps:.6e}"
    )

    # ======================================================
    # MODAL DECAY RATES
    # ======================================================

    df["rate_lines"] = np.nan
    df["rate_fps"] = np.nan

    for i in range(1, len(df)):

        n0 = df.loc[i - 1, "n_modes"]
        n1 = df.loc[i, "n_modes"]

        e0 = df.loc[i - 1, "error_lines"]
        e1 = df.loc[i, "error_lines"]

        df.loc[i, "rate_lines"] = (
            np.log(e0 / e1)
            /
            (n1 - n0)
        )

        e0 = df.loc[i - 1, "error_fps"]
        e1 = df.loc[i, "error_fps"]

        df.loc[i, "rate_fps"] = (
            np.log(e0 / e1)
            /
            (n1 - n0)
        )

    df.to_csv(
        run_dir /
        "modal_comparison.csv",
        index=False,
    )

    # ======================================================
    # ERROR COMPARISON
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.semilogy(
        df["n_modes"],
        df["error_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.semilogy(
        df["n_modes"],
        df["error_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.semilogy(
        Nm,
        fit_lines,
        "--b",
        linewidth=2,
        label=rf"Two-line $Ce^{{-{beta_lines:.3f}N}}$",
    )

    plt.semilogy(
        Nm,
        fit_fps,
        "--r",
        linewidth=2,
        label=rf"FPS $Ce^{{-{beta_fps:.3f}N}}$",
    )

    plt.xlabel(
        "Modes per interface"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "modal_error_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================================
    # CONDITION NUMBER
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.semilogy(
        df["n_modes"],
        df["condition_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.semilogy(
        df["n_modes"],
        df["condition_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Modes per interface"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "modal_condition_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================================
    # OBSERVED BETA
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.plot(
        df["n_modes"][1:],
        df["rate_lines"][1:],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.plot(
        df["n_modes"][1:],
        df["rate_fps"][1:],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.axhline(
        beta_lines,
        linestyle="--",
        color="b",
    )

    plt.axhline(
        beta_fps,
        linestyle="--",
        color="r",
    )

    plt.xlabel(
        "Modes per interface"
    )

    plt.ylabel(
        r"Observed $\beta$"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "modal_rate_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================================
    # LOG(ERROR) FIT
    # ======================================================

    plt.figure(figsize=(8,6))

    plt.plot(
        Nm,
        np.log(df["error_lines"]),
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.plot(
        Nm,
        np.log(df["error_fps"]),
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.plot(
        Nm,
        np.log(fit_lines),
        "--b",
        linewidth=2,
    )

    plt.plot(
        Nm,
        np.log(fit_fps),
        "--r",
        linewidth=2,
    )

    plt.xlabel(
        "Modes per interface"
    )

    plt.ylabel(
        r"$\log(\|e\|)$"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "modal_exponential_fit.png",
        dpi=400,
    )

    plt.close()

    return df


def run_observation_comparison():

    logger.info("=" * 80)

    logger.info(
        "Running observation comparison study "
        "(Two-Line vs FPS)"
    )

    n_modes = 2

    obs_counts = [
        16,
        24,
        32,
        64,
        128,
    ]

    rows = []

    for n_obs_total in obs_counts:

        logger.info(
            f"Observations = {n_obs_total}"
        )

        case = build_case(
            n_modes=n_modes,
            overlap_x_left=8,
            overlap_x_right=8,
            overlap_y_top=8,
            overlap_y_bottom=8,
        )

        U = case["U"]

        basisF = case["basisF"]
        uF = case["uF"]

        Xf = basisF.doflocs[0]
        Yf = basisF.doflocs[1]

        xW = case["xW"]
        xE = case["xE"]

        yS = case["yS"]
        yN = case["yN"]

        Lx = case["Lx"]
        Ly = case["Ly"]

        # ======================================
        # TWO-LINE OBSERVATIONS
        # ======================================

        n_obs_line = max(
            2,
            n_obs_total // 8,
        )

        yD12 = np.linspace(
            0.0,
            yN,
            n_obs_line,
        )

        yD34 = np.linspace(
            yS,
            Ly,
            n_obs_line,
        )

        xD13 = np.linspace(
            0.0,
            xE,
            n_obs_line,
        )

        xD24 = np.linspace(
            xW,
            Lx,
            n_obs_line,
        )

        obs_D12_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD12,
            ]
        )

        obs_D12_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD12,
            ]
        )

        obs_D34_W = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xW,
                ),
                yD34,
            ]
        )

        obs_D34_E = np.column_stack(
            [
                np.full(
                    n_obs_line,
                    xE,
                ),
                yD34,
            ]
        )

        obs_D13_N = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D13_S = np.column_stack(
            [
                xD13,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        obs_D24_N = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yN,
                ),
            ]
        )

        obs_D24_S = np.column_stack(
            [
                xD24,
                np.full(
                    n_obs_line,
                    yS,
                ),
            ]
        )

        constraint_lines = np.vstack(
            [
                obs_D12_W,
                obs_D12_E,

                obs_D34_W,
                obs_D34_E,

                obs_D13_N,
                obs_D13_S,

                obs_D24_N,
                obs_D24_S,
            ]
        )

        # ======================================
        # FPS OBSERVATIONS
        # ======================================

        xx = np.linspace(
            xW,
            xE,
            100,
        )

        yy = np.linspace(
            yS,
            yN,
            100,
        )

        XX, YY = np.meshgrid(
            xx,
            yy,
        )

        candidate_points = np.c_[
            XX.ravel(),
            YY.ravel(),
        ]

        constraint_fps = (
            ObservationStrategy.fps(
                candidate_points,
                n_obs_total,
            )
        )

        # ======================================
        # TWO-LINE
        # ======================================

        result_lines = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_lines,
            )
        )

        interp_lines = LinearNDInterpolator(
            np.c_[
                result_lines.full_x,
                result_lines.full_y,
            ],
            result_lines.full_solution,
            fill_value=np.nan,
        )

        u_lines = interp_lines(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_lines)

        error_lines = np.linalg.norm(
            u_lines[mask]
            -
            uF[mask]
        )

        # ======================================
        # FPS
        # ======================================

        result_fps = (
            OversetInterfaceCorrector2D
            .correct(
                U=U,
                constraint_points=
                constraint_fps,
            )
        )

        interp_fps = LinearNDInterpolator(
            np.c_[
                result_fps.full_x,
                result_fps.full_y,
            ],
            result_fps.full_solution,
            fill_value=np.nan,
        )

        u_fps = interp_fps(
            Xf,
            Yf,
        )

        mask = ~np.isnan(u_fps)

        error_fps = np.linalg.norm(
            u_fps[mask]
            -
            uF[mask]
        )

        rows.append(
            {
                "n_obs":
                n_obs_total,

                "n_line_obs":
                len(constraint_lines),

                "error_lines":
                error_lines,

                "error_fps":
                error_fps,

                "cond_lines":
                result_lines.condition_number,

                "cond_fps":
                result_fps.condition_number,

                "residual_lines":
                result_lines.residual_norm,

                "residual_fps":
                result_fps.residual_norm,
            }
        )

    df = pd.DataFrame(rows)

    df["rate_lines"] = np.nan
    df["rate_fps"] = np.nan

    for i in range(1, len(df)):

        n0 = df.loc[i - 1, "n_obs"]
        n1 = df.loc[i, "n_obs"]

        df.loc[i, "rate_lines"] = (
            np.log(
                df.loc[i - 1, "error_lines"]
                /
                df.loc[i, "error_lines"]
            )
            /
            np.log(n1 / n0)
        )

        df.loc[i, "rate_fps"] = (
            np.log(
                df.loc[i - 1, "error_fps"]
                /
                df.loc[i, "error_fps"]
            )
            /
            np.log(n1 / n0)
        )

    df.to_csv(
        run_dir /
        "observation_comparison.csv",
        index=False,
    )

    # ======================================
    # ERROR
    # ======================================

    plt.figure(figsize=(8,6))

    plt.semilogy(
        df["n_obs"],
        df["error_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.semilogy(
        df["n_obs"],
        df["error_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Number of observations"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "observation_error_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================
    # CONDITIONING
    # ======================================

    plt.figure(figsize=(8,6))

    plt.semilogy(
        df["n_obs"],
        df["cond_lines"],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.semilogy(
        df["n_obs"],
        df["cond_fps"],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Number of observations"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "observation_condition_comparison.png",
        dpi=400,
    )

    plt.close()

    # ======================================
    # OBSERVED RATES
    # ======================================

    plt.figure(figsize=(8,6))

    plt.plot(
        df["n_obs"][1:],
        df["rate_lines"][1:],
        "o-",
        linewidth=2,
        label="Two-line",
    )

    plt.plot(
        df["n_obs"][1:],
        df["rate_fps"][1:],
        "s-",
        linewidth=2,
        label="FPS",
    )

    plt.xlabel(
        "Number of observations"
    )

    plt.ylabel(
        "Observed rate"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir /
        "observation_rate_comparison.png",
        dpi=400,
    )

    plt.close()

    return df

run_modal_comparison()  
#run_overlap_width_comparison()
#single = single_run()  
#run_observation_comparison()
#sing = single_fps_run()
