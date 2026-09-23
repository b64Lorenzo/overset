#!/usr/bin/env python3

"""
overlapping2d.py
================

Validation framework for the 2D overlap reconstruction methodology.

Studies
--------
1. Single reconstruction case
2. Homogeneous reconstruction validation
3. Observation strategy comparison
4. Modal convergence study
5. Mesh convergence study
6. Overlap-width sensitivity study
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from skfem import (
    BilinearForm,
    LinearForm,
)

from skfem.helpers import (
    grad,
    dot,
)

from src.utils.logger import Logger

from src.problems.elliptic2d import (
    Elliptic2DProblem,
    Elliptic2DCoefficients,
)

from src.reconstruction.overlap2d import (
    Domain2D,
    DomainSolution2D,
    SolutionDomain2D,
    OversetInterfaceCorrector2D,
    ObservationStrategy,
)

from src.solver.fem2d import (
    solve_rect,
)

from src.plotlib.solution_plots_2d import (
    plot_reference_solution,
    plot_reconstructed_solution,
)

from src.plotlib.validation_plots_2d import (
    plot_error_before,
    plot_error_after,
    plot_modal_convergence,
    plot_mesh_convergence,
    plot_overlap_study,
    plot_observation_study,
    plot_conditioning,
)
from src.plotlib.overlap_plots_2d import (
    plot_two_line_observations,
    plot_fps_observations,
    plot_overlap_geometry,
)



# ============================================================
# LOGGER
# ============================================================

log_mgr = Logger(
    name="overlap2d",
    run_prefix="overlap2d",
)

logger = log_mgr.get_logger()

run_dir = log_mgr.get_run_dir()

# ============================================================
# PLOTS
# ============================================================

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    }
)

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

Lx = 100.0
Ly = 40.0

x_interface = 64.0

overlap_west = 1
overlap_east = 1

noise_amplitude = 30.0

n_modes_default = 5

n_obs_default = 5

# ============================================================
# REFERENCE MESH
# ============================================================

NX_REFERENCE = 400
NY_REFERENCE = 600

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def bc_west(y):
    return 0.0

def bc_east(y):
    return 0.0

def bc_south(x):
    return 0.0

def bc_north(x):
    return 0.0


# ============================================================
# PDE
# ============================================================

PROBLEM = Elliptic2DProblem(

    Elliptic2DCoefficients(

        k=25.0,

        h0=30.0,

        p0=10.0,
        p_amp=64.0,

        kxP=0.64,
        kyP=0.25,

        A_f=1.0,

        fx_wave=0.25,
        fy_wave=0.15,
    )
)

DIFFUSION_FORM,REACTION_FORM,RHS_FORM = (
    PROBLEM.forms()
)

# ============================================================
# FULL DOMAIN BC
# ============================================================

FULL_BC = {

    "W":
    lambda x, y:
    bc_west(y),

    "E":
    lambda x, y:
    bc_east(y),

    "S":
    lambda x, y:
    bc_south(x),

    "N":
    lambda x, y:
    bc_north(x),
}

# ============================================================
# REFERENCE SOLUTION
# ============================================================

def build_reference_solution(
    nx=NX_REFERENCE,
    ny=NY_REFERENCE,
    ):

    mesh, basis, u = solve_rect(

        xmin=0.0,
        xmax=Lx,

        ymin=0.0,
        ymax=Ly,

        nx=nx,
        ny=ny,

        diffusion_form=DIFFUSION_FORM,
        reaction_form=REACTION_FORM,
        rhs_form=RHS_FORM,

        boundary_conditions=FULL_BC,
    )

    interp = LinearNDInterpolator(

        np.c_[
            basis.doflocs[0],
            basis.doflocs[1],
        ],

        u,

        fill_value=np.nan,
    )

    logger.info(
        f"Reference solution solved "
        f"({nx} x {ny})"
    )

    return (PROBLEM,mesh,basis,u,interp)



# ============================================================
# OVERLAP GEOMETRY
# ============================================================

def overlap_geometry(
    nx_west,
    nx_east,
):

    hx_west = (
        x_interface
        /
        nx_west
    )

    hx_east = (
        Lx
        -
        x_interface
    ) / nx_east

    x_east_overlap = (
        x_interface
        +
        overlap_west
        *
        hx_west
    )

    x_west_overlap = (
        x_interface
        -
        overlap_east
        *
        hx_east
    )

    return (
        x_east_overlap,
        x_west_overlap,
    )


# ============================================================
# OVERLAP TEST CASE
# ============================================================

def build_overlap_case(
    nx_reference=200,
    ny_reference=300,
    nx_west=300,
    nx_east=300,
    n_modes=n_modes_default,
):

    (
        problem,
        mesh_ref,
        basis_ref,
        u_ref,
        interp_ref,
    ) = build_reference_solution(
        nx_reference,
        ny_reference,
    )

    (
        x_east_overlap,
        x_west_overlap,
    ) = overlap_geometry(
        nx_west,
        nx_east,
    )

    # --------------------------------------------------------
    # INTERFACE VALUES FROM REFERENCE
    # --------------------------------------------------------

    y_interface = np.linspace(
        0.0,
        Ly,
        ny_reference + 1,
    )

    u_east_exact = interp_ref(

        np.full_like(
            y_interface,
            x_east_overlap,
        ),

        y_interface,
    )

    u_west_exact = interp_ref(

        np.full_like(
            y_interface,
            x_west_overlap,
        ),

        y_interface,
    )

    u_east_guess = (
        u_east_exact
        +
        noise_amplitude
        *
        np.sin(
            2.0
            *
            np.pi
            *
            y_interface
            /
            Ly
        )
    )

    u_west_guess = (
        u_west_exact
        +
        noise_amplitude
        *
        np.sin(
            -2.0
            *
            np.pi
            *
            y_interface
            /
            Ly
        )
    )

    # --------------------------------------------------------
    # WEST DOMAIN
    # --------------------------------------------------------

    logger.info(
        "Solving West domain"
    )

    west_bc = {

        "W":
        lambda x,y:
        bc_west(y),

        "S":
        lambda x,y:
        bc_south(x),

        "N":
        lambda x,y:
        bc_north(x),

        "E":
        {
            "coords":
            y_interface,

            "values":
            u_east_guess,
        },
    }

    mesh_west, basis_west, u_west = solve_rect(

        xmin=0.0,
        xmax=x_east_overlap,

        ymin=0.0,
        ymax=Ly,

        nx=nx_west,
        ny=ny_reference,

        diffusion_form=DIFFUSION_FORM,
        reaction_form=REACTION_FORM,
        rhs_form=RHS_FORM,

        boundary_conditions=west_bc,
    )

    # --------------------------------------------------------
    # EAST DOMAIN
    # --------------------------------------------------------

    logger.info(
        "Solving East domain"
    )

    east_bc = {

        "W":
        {
            "coords":
            y_interface,

            "values":
            u_west_guess,
        },

        "E":
        lambda x,y:
        bc_east(y),

        "S":
        lambda x,y:
        bc_south(x),

        "N":
        lambda x,y:
        bc_north(x),
    }

    mesh_east, basis_east, u_east = solve_rect(

        xmin=x_west_overlap,
        xmax=Lx,

        ymin=0.0,
        ymax=Ly,

        nx=nx_east,
        ny=ny_reference,

        diffusion_form=DIFFUSION_FORM,
        reaction_form=REACTION_FORM,
        rhs_form=RHS_FORM,

        boundary_conditions=east_bc,
    )

    # --------------------------------------------------------
    # INTERPOLATORS
    # --------------------------------------------------------

    interp_west = LinearNDInterpolator(

        np.c_[
            basis_west.doflocs[0],
            basis_west.doflocs[1],
        ],

        u_west,

        fill_value=np.nan,
    )

    interp_east = LinearNDInterpolator(

        np.c_[
            basis_east.doflocs[0],
            basis_east.doflocs[1],
        ],

        u_east,

        fill_value=np.nan,
    )

    # --------------------------------------------------------
    # BLENDING WEIGHTS
    # --------------------------------------------------------

    west_weight = (
        lambda x,y:
        (
            x_east_overlap - x
        )
        /
        (
            x_east_overlap
            -
            x_west_overlap
        )
    )

    east_weight = (
        lambda x,y:
        (
            x
            -
            x_west_overlap
        )
        /
        (
            x_east_overlap
            -
            x_west_overlap
        )
    )

    # --------------------------------------------------------
    # DOMAIN OBJECTS
    # --------------------------------------------------------

    domain_west = Domain2D(

        a=0.0,
        b=x_east_overlap,

        c=0.0,
        d=Ly,

        F0=30.0,

        k=problem.k,

        interfaces=("E",),

        n_modes=n_modes,

        weight_function=
        west_weight,
    )

    domain_east = Domain2D(

        a=x_west_overlap,
        b=Lx,

        c=0.0,
        d=Ly,

        F0=30.0,

        k=problem.k,

        interfaces=("W",),

        n_modes=n_modes,

        weight_function=
        east_weight,
    )

    D_west = DomainSolution2D(

        domain=domain_west,

        solution=u_west,

        x=basis_west.doflocs[0],

        y=basis_west.doflocs[1],

        interpolator=interp_west,
    )

    D_east = DomainSolution2D(

        domain=domain_east,

        solution=u_east,

        x=basis_east.doflocs[0],

        y=basis_east.doflocs[1],

        interpolator=interp_east,
    )

    U = SolutionDomain2D(
        domains=(
            D_west,
            D_east,
        )
    )

    return {

        "problem":
        problem,

        "solution_domain":
        U,

        "reference_mesh":
        mesh_ref,

        "reference_basis":
        basis_ref,

        "reference_solution":
        u_ref,

        "reference_interpolator":
        interp_ref,

        "basis_west":
        basis_west,

        "basis_east":
        basis_east,

        "mesh_west":
        mesh_west,

        "mesh_east":
        mesh_east,

        "u_west":
        u_west,

        "u_east":
        u_east,

        "interp_west":
        interp_west,

        "interp_east":
        interp_east,

        "x_west_overlap":
        x_west_overlap,

        "x_east_overlap":
        x_east_overlap,

        "y_interface":
        y_interface,
    }


# ============================================================
# OBSERVATION STRATEGIES
# ============================================================

def build_two_line_observations(
    x_west_overlap,
    x_east_overlap,
    n_obs=n_obs_default,
):

    return (
        ObservationStrategy
        .two_lines(

            xW=x_west_overlap,

            xE=x_east_overlap,

            ymin=0.0,

            ymax=Ly,

            n_y=n_obs,
        )
    )


def build_fps_observations(
    x_west_overlap,
    x_east_overlap,
    n_points=40,
):

    xx = np.linspace(
        x_west_overlap,
        x_east_overlap,
        60,
    )

    yy = np.linspace(
        0.0,
        Ly,
        60,
    )

    XX, YY = np.meshgrid(
        xx,
        yy,
    )

    candidates = np.c_[
        XX.ravel(),
        YY.ravel(),
    ]

    return (
        ObservationStrategy
        .fps(
            candidates,
            n_points,
        )
    )


# ============================================================
# RECONSTRUCTION
# ============================================================

def run_reconstruction(
    U,
    observation_points,
):

    logger.info(
        f"Observation points = "
        f"{len(observation_points)}"
    )

    logger.info(
        f"Unknown amplitudes = "
        f"{U.total_unknowns}"
    )

    return (
        OversetInterfaceCorrector2D
        .correct(
            U=U,
            constraint_points=
            observation_points,
        )
    )


# ============================================================
# RECONSTRUCTED INTERPOLATOR
# ============================================================

def build_reconstructed_interpolator(
    result,
):
    """
    Interpolator of the reconstructed
    global overset solution.
    """

    return LinearNDInterpolator(

        np.c_[
            result.full_x,
            result.full_y,
        ],

        result.full_solution,

        fill_value=np.nan,
    )


# ============================================================
# UNCORRECTED INTERPOLATOR
# ============================================================

def build_uncorrected_interpolator(
    case,
):
    """
    Interpolator built directly from the
    uncorrected West/East solutions.
    """

    basis_west = case["basis_west"]
    basis_east = case["basis_east"]

    u_west = case["u_west"]
    u_east = case["u_east"]

    return LinearNDInterpolator(

        np.c_[

            np.concatenate([
                basis_west.doflocs[0],
                basis_east.doflocs[0],
            ]),

            np.concatenate([
                basis_west.doflocs[1],
                basis_east.doflocs[1],
            ]),
        ],

        np.concatenate([
            u_west,
            u_east,
        ]),

        fill_value=np.nan,
    )


# ============================================================
# GLOBAL L2 ERROR
# ============================================================

def compute_global_l2_error(
    result,
    basis_ref,
    interp_ref,
):
    """
    Compute global reconstruction error
    against the reference solution.
    """

    X = basis_ref.doflocs[0]
    Y = basis_ref.doflocs[1]

    u_ref = interp_ref(
        X,
        Y,
    )

    interp_rec = (
        build_reconstructed_interpolator(
            result
        )
    )

    u_rec = interp_rec(
        X,
        Y,
    )

    mask = (
        ~np.isnan(u_ref)
        &
        ~np.isnan(u_rec)
    )

    if np.sum(mask) == 0:

        return np.nan

    return np.linalg.norm(
        u_rec[mask]
        -
        u_ref[mask]
    )


# ============================================================
# SINGLE CASE
# ============================================================

def run_single_case():

    logger.info(
        "=" * 80
    )

    logger.info(
        "Running single-case reconstruction"
    )

    case = build_overlap_case()

    U = case["solution_domain"]

    basis_ref = case["reference_basis"]

    interp_ref = case[
        "reference_interpolator"
    ]

    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    observation_points = (
        build_two_line_observations(
            case["x_west_overlap"],
            case["x_east_overlap"],
        )
    )

    # --------------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------------

    result = run_reconstruction(
        U,
        observation_points,
    )

    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    Xref = basis_ref.doflocs[0]
    Yref = basis_ref.doflocs[1]

    u_ref = case[
        "reference_solution"
    ]

    # --------------------------------------------------------
    # RECONSTRUCTED
    # --------------------------------------------------------

    interp_rec = (
        build_reconstructed_interpolator(
            result
        )
    )

    u_after = interp_rec(
        Xref,
        Yref,
    )

    # --------------------------------------------------------
    # UNCORRECTED
    # --------------------------------------------------------

    interp_unc = (
        build_uncorrected_interpolator(
            case
        )
    )

    u_before = interp_unc(
        Xref,
        Yref,
    )

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    mask_before = (
        ~np.isnan(u_before)
        &
        ~np.isnan(u_ref)
    )

    mask_after = (
        ~np.isnan(u_after)
        &
        ~np.isnan(u_ref)
    )

    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    err_before = np.zeros_like(
        u_ref
    )

    err_after = np.zeros_like(
        u_ref
    )

    err_before[mask_before] = (
        u_before[mask_before]
        -
        u_ref[mask_before]
    )

    err_after[mask_after] = (
        u_after[mask_after]
        -
        u_ref[mask_after]
    )

    L2_before = np.linalg.norm(
        err_before[mask_before]
    )

    L2_after = np.linalg.norm(
        err_after[mask_after]
    )

    improvement = (
        L2_before
        /
        L2_after
    )

    logger.info(
        f"L2 before = "
        f"{L2_before:.6e}"
    )

    logger.info(
        f"L2 after = "
        f"{L2_after:.6e}"
    )

    logger.info(
        f"Improvement = "
        f"{improvement:.3f}"
    )

    # --------------------------------------------------------
    # RECONSTRUCTION DIAGNOSTICS
    # --------------------------------------------------------

    if hasattr(
        result,
        "condition_number",
    ):

        logger.info(
            f"Condition number = "
            f"{result.condition_number:.6e}"
        )

    if hasattr(
        result,
        "residual_norm",
    ):

        logger.info(
            f"Residual norm = "
            f"{result.residual_norm:.6e}"
        )

    # ==========================================================
    # PLOTS
    # ==========================================================

    plot_two_line_observations(
        observation_points,
        run_dir,
    )

    plot_reference_solution(
        Xref,
        Yref,
        u_ref,
        run_dir,
    )

    plot_reconstructed_solution(
        result.full_x,
        result.full_y,
        result.full_solution,
        run_dir,
    )

    plot_error_before(
        Xref,
        Yref,
        err_before,
        run_dir,
    )

    plot_error_after(
        Xref,
        Yref,
        err_after,
        run_dir,
    )


# ============================================================
# HOMOGENEOUS VALIDATION
# ============================================================

def run_homogeneous_validation():

    logger.info("=" * 80)
    logger.info("Running homogeneous validation")

    case = build_overlap_case(
        n_modes=8,
    )

    U = case["solution_domain"]

    obs = build_two_line_observations(
        case["x_west_overlap"],
        case["x_east_overlap"],
    )

    result = run_reconstruction(
        U,
        obs,
    )

    e_west = result.corrections[0]
    e_east = result.corrections[1]

    basis_west = case["basis_west"]
    basis_east = case["basis_east"]

    XW = basis_west.doflocs[0]
    YW = basis_west.doflocs[1]

    XE = basis_east.doflocs[0]
    YE = basis_east.doflocs[1]

    interp_ref = case[
        "reference_interpolator"
    ]

    u_ref_west = interp_ref(
        XW,
        YW,
    )

    u_ref_east = interp_ref(
        XE,
        YE,
    )

    true_h_west = (
        case["u_west"]
        -
        u_ref_west
    )

    true_h_east = (
        case["u_east"]
        -
        u_ref_east
    )

    err_west = np.linalg.norm(
        true_h_west - e_west
    )

    err_east = np.linalg.norm(
        true_h_east - e_east
    )

    logger.info(
        f"West homogeneous error = {err_west:.6e}"
    )

    logger.info(
        f"East homogeneous error = {err_east:.6e}"
    )

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XW,
        YW,
        true_h_west - e_west,
        levels=100,
        cmap="seismic",
    )

    plt.colorbar()

    plt.title(
        "West Homogeneous Reconstruction Error"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "homogeneous_error_west.png",
        dpi=400,
    )

    plt.close()

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XE,
        YE,
        true_h_east - e_east,
        levels=100,
        cmap="seismic",
    )

    plt.colorbar()

    plt.title(
        "East Homogeneous Reconstruction Error"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "homogeneous_error_east.png",
        dpi=400,
    )

    plt.close()

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XW,
        YW,
        true_h_west,
        levels=100,
        cmap="viridis",
    )

    plt.colorbar()

    plt.title(
        "True West Homogeneous Correction"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "true_homogeneous_west.png",
        dpi=400,
    )

    plt.close()

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XE,
        YE,
        true_h_east,
        levels=100,
        cmap="viridis",
    )

    plt.colorbar()

    plt.title(
        "True East Homogeneous Correction"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "true_homogeneous_east.png",
        dpi=400,
    )

    plt.close()

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XW,
        YW,
        e_west,
        levels=100,
        cmap="viridis",
    )

    plt.colorbar()

    plt.title(
        "Reconstructed West Homogeneous Correction"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "reconstructed_homogeneous_west.png",
        dpi=400,
    )

    plt.close()

    plt.figure(figsize=(11,7))

    plt.tricontourf(
        XE,
        YE,
        e_east,
        levels=100,
        cmap="viridis",
    )

    plt.colorbar()

    plt.title(
        "Reconstructed East Homogeneous Correction"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "reconstructed_homogeneous_east.png",
        dpi=400,
    )

    plt.close()

    return {

        "west":
        err_west,

        "east":
        err_east,

        "true_h_west":
        true_h_west,

        "true_h_east":
        true_h_east,

        "reconstructed_h_west":
        e_west,

        "reconstructed_h_east":
        e_east,
    }

# ============================================================
# MODAL / OBSERVATION CONVERGENCE
# ============================================================

def run_modal_observation_convergence():

    logger.info("=" * 80)
    logger.info(
        "Running modal-observation convergence study"
    )

    mode_counts = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]

    rows = []

    for n_modes in mode_counts:

        logger.info(
            f"Modes/domain = {n_modes}"
        )

        case = build_overlap_case(
            n_modes=n_modes,
        )

        U = case["solution_domain"]

        obs = build_two_line_observations(
            case["x_west_overlap"],
            case["x_east_overlap"],
            n_obs=n_modes,
        )

        result = run_reconstruction(
            U,
            obs,
        )

        error = compute_global_l2_error(
            result,
            case["reference_basis"],
            case["reference_interpolator"],
        )

        rows.append(
            {
                "n_modes": n_modes,
                "n_obs_total": len(obs),
                "n_unknowns": U.total_unknowns,
                "error": error,
                "cond": result.condition_number,
                "residual":
                result.residual_norm,
            }
        )

        logger.info(
            f"Error = {error:.6e}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        run_dir
        /
        "modal_observation_convergence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["n_modes"],
        df["error"],
        "o-",
        linewidth=2,
        label="L2 error",
    )

    plt.xlabel(
        "Modes per domain = observations per line"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(True, which="both")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "modal_observation_convergence.png",
        dpi=400,
    )

    plt.close()

    # --------------------------------------------------------
    # CONDITIONING
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["n_modes"],
        df["cond"],
        "o-r",
        linewidth=2,
    )

    plt.xlabel(
        "Modes per domain"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "modal_observation_conditioning.png",
        dpi=400,
    )

    plt.close()

    return df


# ============================================================
# OVERLAP WIDTH CONVERGENCE
# ============================================================

def run_overlap_width_convergence():

    logger.info("=" * 80)
    logger.info(
        "Running overlap-width convergence"
    )

    overlap_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ]

    rows = []

    for overlap in overlap_sizes:

        logger.info(
            f"Overlap cells = {overlap}"
        )

        case = build_overlap_case(

            n_modes=5,

            overlap_w=overlap,
            overlap_e=overlap,
        )

        U = case["solution_domain"]

        obs = build_two_line_observations(

            case["x_west_overlap"],
            case["x_east_overlap"],

            n_obs=5,
        )

        result = run_reconstruction(
            U,
            obs,
        )

        error = compute_global_l2_error(

            result,

            case["reference_basis"],
            case["reference_interpolator"],
        )

        overlap_width = (
            case["x_east_overlap"]
            -
            case["x_west_overlap"]
        )

        rows.append(
            {
                "overlap_cells":
                overlap,

                "overlap_width":
                overlap_width,

                "error":
                error,

                "cond":
                result.condition_number,
            }
        )

        logger.info(
            f"Width = {overlap_width:.6f}"
        )

        logger.info(
            f"Error = {error:.6e}"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        run_dir
        /
        "overlap_width_study.csv",
        index=False,
    )

    # ----------------------------------------------
    # ERROR
    # ----------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["overlap_width"],
        df["error"],
        "o-",
        linewidth=2,
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "overlap_width_error.png",
        dpi=400,
    )

    plt.close()

    # ----------------------------------------------
    # CONDITIONING
    # ----------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["overlap_width"],
        df["cond"],
        "o-r",
        linewidth=2,
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "overlap_width_conditioning.png",
        dpi=400,
    )

    plt.close()

    return df


# ============================================================
# OVERLAP CASE WITH VARIABLE OVERLAP WIDTH
# ============================================================

def build_overlap_width_case(
    overlap_cells,
    nx_reference=200,
    ny_reference=300,
    nx_west=300,
    nx_east=300,
    n_modes=5,
):

    (
        problem,
        mesh_ref,
        basis_ref,
        u_ref,
        interp_ref,
    ) = build_reference_solution(
        nx_reference,
        ny_reference,
    )

    hx_west = (
        x_interface
        /
        nx_west
    )

    hx_east = (
        Lx
        -
        x_interface
    ) / nx_east

    x_east_overlap = (
        x_interface
        +
        overlap_cells
        *
        hx_west
    )

    x_west_overlap = (
        x_interface
        -
        overlap_cells
        *
        hx_east
    )

    y_interface = np.linspace(
        0.0,
        Ly,
        ny_reference + 1,
    )

    u_east_exact = interp_ref(
        np.full_like(
            y_interface,
            x_east_overlap,
        ),
        y_interface,
    )

    u_west_exact = interp_ref(
        np.full_like(
            y_interface,
            x_west_overlap,
        ),
        y_interface,
    )

    u_east_guess = (
        u_east_exact
        +
        noise_amplitude
        *
        np.sin(
            2.0
            *
            np.pi
            *
            y_interface
            /
            Ly
        )
    )

    u_west_guess = (
        u_west_exact
        +
        noise_amplitude
        *
        np.sin(
            -2.0
            *
            np.pi
            *
            y_interface
            /
            Ly
        )
    )

    west_bc = {

        "W":
        lambda x,y:
        bc_west(y),

        "S":
        lambda x,y:
        bc_south(x),

        "N":
        lambda x,y:
        bc_north(x),

        "E":
        {
            "coords":
            y_interface,

            "values":
            u_east_guess,
        },
    }

    east_bc = {

        "W":
        {
            "coords":
            y_interface,

            "values":
            u_west_guess,
        },

        "E":
        lambda x,y:
        bc_east(y),

        "S":
        lambda x,y:
        bc_south(x),

        "N":
        lambda x,y:
        bc_north(x),
    }

    mesh_west, basis_west, u_west = solve_rect(

        xmin=0.0,
        xmax=x_east_overlap,

        ymin=0.0,
        ymax=Ly,

        nx=nx_west,
        ny=ny_reference,

        diffusion_form=DIFFUSION_FORM,
        reaction_form=REACTION_FORM,
        rhs_form=RHS_FORM,

        boundary_conditions=west_bc,
    )

    mesh_east, basis_east, u_east = solve_rect(

        xmin=x_west_overlap,
        xmax=Lx,

        ymin=0.0,
        ymax=Ly,

        nx=nx_east,
        ny=ny_reference,

        diffusion_form=DIFFUSION_FORM,
        reaction_form=REACTION_FORM,
        rhs_form=RHS_FORM,

        boundary_conditions=east_bc,
    )

    interp_west = LinearNDInterpolator(

        np.c_[
            basis_west.doflocs[0],
            basis_west.doflocs[1],
        ],

        u_west,

        fill_value=np.nan,
    )

    interp_east = LinearNDInterpolator(

        np.c_[
            basis_east.doflocs[0],
            basis_east.doflocs[1],
        ],

        u_east,

        fill_value=np.nan,
    )

    west_weight = (
        lambda x,y:
        (
            x_east_overlap - x
        )
        /
        (
            x_east_overlap
            -
            x_west_overlap
        )
    )

    east_weight = (
        lambda x,y:
        (
            x
            -
            x_west_overlap
        )
        /
        (
            x_east_overlap
            -
            x_west_overlap
        )
    )

    domain_west = Domain2D(

        a=0.0,
        b=x_east_overlap,

        c=0.0,
        d=Ly,

        F0=30.0,

        k=problem.k,

        interfaces=("E",),

        n_modes=n_modes,

        weight_function=
        west_weight,
    )

    domain_east = Domain2D(

        a=x_west_overlap,
        b=Lx,

        c=0.0,
        d=Ly,

        F0=30.0,

        k=problem.k,

        interfaces=("W",),

        n_modes=n_modes,

        weight_function=
        east_weight,
    )

    D_west = DomainSolution2D(

        domain=domain_west,

        solution=u_west,

        x=basis_west.doflocs[0],
        y=basis_west.doflocs[1],

        interpolator=interp_west,
    )

    D_east = DomainSolution2D(

        domain=domain_east,

        solution=u_east,

        x=basis_east.doflocs[0],
        y=basis_east.doflocs[1],

        interpolator=interp_east,
    )

    U = SolutionDomain2D(
        domains=(
            D_west,
            D_east,
        )
    )

    return {

        "solution_domain":
        U,

        "reference_basis":
        basis_ref,

        "reference_interpolator":
        interp_ref,

        "x_west_overlap":
        x_west_overlap,

        "x_east_overlap":
        x_east_overlap,
    }


# ============================================================
# OVERLAP WIDTH CONVERGENCE
# ============================================================

def run_overlap_width_convergence():

    logger.info("=" * 80)

    logger.info(
        "Running overlap-width convergence study"
    )

    overlap_sizes = [
        1,
        10,
        100,
        150
    ]

    rows = []

    for overlap in overlap_sizes:

        logger.info(
            f"Overlap cells = {overlap}"
        )

        case = (
            build_overlap_width_case(
                overlap_cells=overlap,
                n_modes=5,
            )
        )

        U = case["solution_domain"]

        obs = build_two_line_observations(

            case["x_west_overlap"],
            case["x_east_overlap"],

            n_obs=5,
        )

        result = run_reconstruction(
            U,
            obs,
        )

        error = compute_global_l2_error(

            result,

            case["reference_basis"],

            case["reference_interpolator"],
        )

        width = (
            case["x_east_overlap"]
            -
            case["x_west_overlap"]
        )

        rows.append(
            {
                "overlap_cells":
                overlap,

                "overlap_width":
                width,

                "error":
                error,

                "condition":
                result.condition_number,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        run_dir
        /
        "overlap_width_convergence.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["overlap_width"],
        df["error"],
        "o-",
        linewidth=2,
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Global L2 error"
    )

    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "overlap_width_error.png",
        dpi=400,
    )

    plt.close()

    # --------------------------------------------------------
    # CONDITION NUMBER
    # --------------------------------------------------------

    plt.figure(figsize=(8,6))

    plt.loglog(
        df["overlap_width"],
        df["condition"],
        "o-r",
        linewidth=2,
    )

    plt.xlabel(
        "Overlap width"
    )

    plt.ylabel(
        "Condition number"
    )

    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "overlap_width_conditioning.png",
        dpi=400,
    )

    plt.close()

    return df

# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 80)
    logger.info("Starting overlap2d validation")
    logger.info("=" * 80)

    single_case = run_single_case()

    #homogeneous = run_homogeneous_validation()

    #overlap_width = run_overlap_width_convergence()

    #modal_observation = run_modal_observation_convergence()

    logger.info("=" * 80)
    logger.info("Finished overlap2d validation")
    logger.info("=" * 80)

    print()
    print(
        "Overlap2D validation completed."
    )

    print(
        f"Results saved in {run_dir}"
    )

    print()


if __name__ == "__main__":

    main()