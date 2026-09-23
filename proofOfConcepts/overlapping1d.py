"""
overlapping1d.py
================

Author
------
Lorenzo Zambelli

Version
-------
2.0

Description
-----------
Proof-of-concept implementation of the one-dimensional overlapping-domain
reconstruction methodology developed in this work.

The script reproduces the validation experiments described in the
"Overlapping Domains - 1D Case" section of the report and serves as the
main entry point for the numerical proof-of-concept study.

The objective is to verify that the homogeneous correction introduced by
artificial domain decomposition can be reconstructed from overlap
observations and subsequently removed from the independently computed
subdomain solutions. The reconstructed solution is then compared against
a benchmark solution computed on the original undecomposed domain.

Mathematical Problem
--------------------
The governing elliptic problem is

    k u - d/dx(h du/dx)
        =
        d/dx(p df/dx),

defined on the computational domain

    Ω = [x0, xN].

The domain is partitioned into two overlapping subdomains

    ΩW = [x0, xE],
    ΩE = [xW, xN],

with observation region

    Ωobs = [xW, xE].

The solution is decomposed as

    u = up + uh,

where

    up : particular solution,
    uh : homogeneous correction.

The reconstruction procedure exploits the assumption that the mismatch
between the subdomain solutions is dominated by the homogeneous
corrections introduced by the artificial interface conditions.

For constant coefficients, the homogeneous problem

    -(h u_x)_x + k u = 0

admits solutions of the form

    exp(±λx),

with

    λ = sqrt(k / h).

These homogeneous modes are reconstructed from overlap observations and
removed from the local subdomain solutions.

Validation Objectives
---------------------
1. Solve the governing equation on the full computational domain.
2. Solve the equation independently on overlapping subdomains.
3. Construct overlap observations.
4. Estimate homogeneous interface amplitudes.
5. Reconstruct the overlap correction.
6. Remove the reconstructed homogeneous contribution.
7. Blend corrected solutions within the overlap region.
8. Compare against the benchmark solution.
9. Evaluate reconstruction accuracy.
10. Generate publication-quality figures and diagnostics.

Repository Context
------------------
This script is intentionally lightweight and delegates most of the
mathematical and numerical operations to dedicated modules:

    src.problems.elliptic1d
        Definition of the governing elliptic problem.

    src.solver.fem1d
        Finite-element discretization and solution of the PDE.

    src.reconstruction.overlap1d
        Construction and solution of the overlap reconstruction system.

    src.plotlib.solution_plots
        Solution-comparison figures.

    src.plotlib.overlap_plots
        Overlap-region diagnostics.

    src.plotlib.validation_plots
        Error, residual, and convergence plots.

    src.utils.logger
        Logging and run-directory management.

Workflow
--------
The numerical experiment proceeds according to the following sequence.

(1) Physical coefficients and forcing terms are defined.

(2) An Elliptic1DProblem instance is created.

(3) A benchmark solution is computed on the entire domain Ω.

(4) Independent finite-element solutions are computed on

        ΩW,
        ΩE.

(5) Observations inside

        Ωobs

    are extracted.

(6) The reconstruction system

        H a = d

    is assembled, where

        H : reconstruction matrix,
        a : homogeneous amplitudes,
        d : overlap mismatch vector.

(7) The amplitudes

        aW,
        aE

    are recovered.

(8) Homogeneous corrections

        uhW,
        uhE

    are reconstructed and removed from the local solutions.

(9) Corrected solutions are blended using a partition-of-unity
    weighting function.

(10) Reconstruction errors, residuals, and diagnostic quantities
     are evaluated and visualized.

Main Parameters
---------------
Lx : float
    Physical domain length.

xI : float
    Interface location.

xW : float
    Western boundary of overlap region.

xE : float
    Eastern boundary of overlap region.

n_full : int
    Number of elements used for the benchmark solution.

n_west : int
    Number of finite elements used on ΩW.

n_east : int
    Number of finite elements used on ΩE.

k : float
    Reaction coefficient.

h(x) : callable
    Water-depth coefficient.

p(x) : callable
    Pressure-related coefficient.

f(x,t) : callable
    Free-surface-related quantity.

Outputs
-------
The script generates

* benchmark solutions,
* reconstructed solutions,
* overlap diagnostics,
* correction profiles,
* residual information,
* error metrics,
* publication-quality figures,
* CSV result files,
* logging information.

Generated Figures
-----------------
solution_with_error.png
    Comparison between exact, reconstructed, and uncorrected solutions.

overlap_detail.png
    Detailed view of the overlap region.

homogeneous_correction.png
    Reconstructed homogeneous modes.

reconstruction_residual.png
    Residual diagnostics.

Generated Data
--------------
results.csv
    Benchmark and reconstructed solution values.

run.log
    Complete execution log.

Typical Usage
-------------
Run directly from the repository root

    python proofOfConcepts/overlapping1d.py

or

    python -m proofOfConcepts.overlapping1d

Expected Outcome
----------------
A successful reconstruction should produce

    ||u_rec - u_exact||
        <
    ||u_unc - u_exact||,

demonstrating that the interface-induced homogeneous error has been
identified and removed.

References
----------
[1] Lawrence, G. A., et al.
    Variational Boussinesq Model.

[2] Klopman, G.
    Variational Boussinesq Model for Linear Water Waves on
    Varying Depth and Current: Numerical Approach.

[3] Olver, P. J.
    Introduction to Partial Differential Equations.
    Springer, 2014.

[4] Smith, B. F., Bjørstad, P., Gropp, W. D.
    Domain Decomposition: Parallel Multilevel Methods
    for Elliptic Partial Differential Equations.
    Cambridge University Press, 1996.

[5] Chesshire, G. and Henshaw, W. D.
    Composite Overlapping Meshes for the Solution of
    Partial Differential Equations.
    Journal of Computational Physics, 1990.

[6] Roos, H.-G., Stynes, M., Tobiska, L.
    Robust Numerical Methods for Singularly Perturbed
    Differential Equations.
    Springer, 2008.

[7] Zambelli, L.
    Hyperbolic Reconstruction Techniques for Overlapping
    and Disjoint Computational Domains in the Variational
    Boussinesq Model.
    MSc Thesis / Research Report, 2026.

Scientific Context
------------------
This proof-of-concept implements the one-dimensional overlap
reconstruction methodology developed in

    Zambelli (2026),

where the interface error introduced by artificial domain
decomposition is represented as a homogeneous correction and
recovered through local observations inside the overlap region.

The formulation builds upon the Variational Boussinesq Model
developed by Lawrence et al. and Klopman, while drawing on
classical results from elliptic boundary-value problems,
eigenfunction expansions, and domain decomposition methods.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.problems.elliptic1d import (
    Elliptic1DProblem,
)

from src.solver.fem1d import (
    FEMEllipticSolver1D,
)

from src.reconstruction.overlap1d import (
    OverlapReconstructor1D,
)

from src.plotlib.solution_plots import (
    plot_solution_with_error,
    plot_sharp_reconstruction,
    plot_weight_blending_comparison,
    plot_global_error_comparison,
)

from src.plotlib.overlap_plots import (
    plot_interface_bc_comparison,
    plot_overlap_detail,
    plot_homogeneous_correction,
    plot_particular_solution_mismatch,
)

from src.plotlib.validation_plots import (
    plot_reconstruction_diagnostics,
    plot_convergence,
    plot_condition_vs_overlap,
    plot_error_vs_overlap,
    plot_error_vs_condition,
)

from src.utils.logger import Logger


# ============================================================
# PLOTTING
# ============================================================

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    }
)


# ============================================================
# LOGGER
# ============================================================

log_mgr = Logger(
    name="overlap1d",
    run_prefix="overlap1d",
)

logger = log_mgr.get_logger()

run_dir = log_mgr.get_run_dir()


# ============================================================
# USER PARAMETERS
# ============================================================

Lx = 100.0

xI = 64.0

n_full = 800

n_west = 400
n_east = 400

delta_west = 1
delta_east = 1

noise_factor = 30.0

current_time = 0.0

k_reaction = 25.0


# ============================================================
# COEFFICIENTS
# ============================================================

h0 = 30.0

p0 = 10.0

p_amp = 64.0

kp = 0.64

Af = 1.0

kf = 0.25

omega = 0.64


# ============================================================
# PHYSICAL FUNCTIONS
# ============================================================

def h(x):

    return (
        h0
        +
        5.0 * np.cos(
            0.64 * x
        )
    )


def h_x(x):

    return (
        -5.0
        * 0.64
        * np.sin(
            0.64 * x
        )
    )


def p(x):

    return (
        p0
        +
        p_amp
        *
        np.cos(
            kp * x
        )
    )


def p_x(x):

    return (
        -p_amp
        *
        kp
        *
        np.sin(
            kp * x
        )
    )


def f_x(x, t):

    return (
        Af
        *
        kf
        *
        np.cos(
            kf * x
            -
            omega * t
        )
    )


def f_xx(x, t):

    return (
        -Af
        *
        kf**2
        *
        np.sin(
            kf * x
            -
            omega * t
        )
    )


# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def west_bc(_):

    return 0.0


def east_bc(_):

    return 0.0


# ============================================================
# PROBLEM
# ============================================================

problem = Elliptic1DProblem(
    k=k_reaction,
    h=h,
    h_x=h_x,
    p=p,
    p_x=p_x,
    f_x=f_x,
    f_xx=f_xx,
)

solver = FEMEllipticSolver1D(
    problem
)

reconstructor = (
    OverlapReconstructor1D()
)


def run_single_case(
    n_full=n_full,
    n_west=n_west,
    n_east=n_east,
    delta_west=delta_west,
    delta_east=delta_east,
    generate_plots=True,
):
    """
    Execute a single overlap-reconstruction experiment.

    Parameters
    ----------
    n_full : int
        Full-domain mesh size.

    n_west : int
        West-domain mesh size.

    n_east : int
        East-domain mesh size.

    delta_west : int
        West overlap thickness in cells.

    delta_east : int
        East overlap thickness in cells.

    generate_plots : bool
        Generate figures.

    Returns
    -------
    dict
        Reconstruction diagnostics.
    """

    # ========================================================
    # GEOMETRY
    # ========================================================

    dx_west = xI / n_west

    dx_east = (
        Lx - xI
    ) / n_east

    xE = (
        xI
        + delta_west * dx_west
    )

    xW = (
        xI
        - delta_east * dx_east
    )

    # ========================================================
    # BENCHMARK
    # ========================================================

    x_full, u_full = solver.solve(
        a=0.0,
        b=Lx,
        n_elements=n_full,
        bc_left=0.0,
        bc_right=0.0,
        current_time=current_time,
    )

    # ========================================================
    # INTERFACE VALUES
    # ========================================================

    gW_exact = np.interp(
        xE,
        x_full,
        u_full,
    )

    gE_exact = np.interp(
        xW,
        x_full,
        u_full,
    )

    gW = (
        noise_factor
        * gW_exact
    )

    gE = (
        noise_factor
        * gE_exact
    )

    if generate_plots:

        plot_interface_bc_comparison(
            x_full=x_full,
            u_full=u_full,
            xW=xW,
            xE=xE,
            gW_exact=gW_exact,
            gE_exact=gE_exact,
            gW=gW,
            gE=gE,
            run_dir=run_dir,
        )

    # ========================================================
    # SUBDOMAIN SOLUTIONS
    # ========================================================

    x_west, u_west = solver.solve(
        a=0.0,
        b=xE,
        n_elements=n_west,
        bc_left=0.0,
        bc_right=gW,
        current_time=current_time,
    )

    x_east, u_east = solver.solve(
        a=xW,
        b=Lx,
        n_elements=n_east,
        bc_left=gE,
        bc_right=0.0,
        current_time=current_time,
    )

    # ========================================================
    # OVERLAP OBSERVATIONS
    # ========================================================

    x_obs = np.array(
        [
            xW,
            xE,
        ]
    )

    uW_obs = np.interp(
        x_obs,
        x_west,
        u_west,
    )

    uE_obs = np.interp(
        x_obs,
        x_east,
        u_east,
    )

    # ========================================================
    # DECAY PARAMETER
    # ========================================================

    lam = (
        problem.averaged_decay_parameter(
            xW,
            xE,
        )
    )

    # ========================================================
    # RECONSTRUCTION
    # ========================================================

    result = reconstructor.solve(
        x_obs=x_obs,
        uW_obs=uW_obs,
        uE_obs=uE_obs,
        x0=0.0,
        xN=Lx,
        xW=xW,
        xE=xE,
        lam=lam,
    )

    a_W = result.amplitudes[0]
    a_E = result.amplitudes[1]

    # ========================================================
    # HOMOGENEOUS MODES
    # ========================================================

    phi_W = reconstructor.phi_west(
        x_west,
        0.0,
        xE,
        lam,
    )

    phi_E = reconstructor.phi_east(
        x_east,
        Lx,
        xW,
        lam,
    )

    u_hW = a_W * phi_W

    u_hE = a_E * phi_E

    # ========================================================
    # CORRECT SOLUTIONS
    # ========================================================

    u_west_corr = (
        u_west
        -
        u_hW
    )

    u_east_corr = (
        u_east
        -
        u_hE
    )

    # ========================================================
    # REFERENCE SUBDOMAINS
    # ========================================================

    u_exact_west = np.interp(
        x_west,
        x_full,
        u_full,
    )

    u_exact_east = np.interp(
        x_east,
        x_full,
        u_full,
    )

    true_hW = (
        u_west
        -
        u_exact_west
    )

    true_hE = (
        u_east
        -
        u_exact_east
    )

    # ========================================================
    # HOMOGENEOUS VALIDATION
    # ========================================================

    if generate_plots:

        plot_homogeneous_correction(
            x_west=x_west,
            true_west=true_hW,
            reconstructed_west=u_hW,
            x_east=x_east,
            true_east=true_hE,
            reconstructed_east=u_hE,
            run_dir=run_dir,
        )

    # ========================================================
    # PARTICULAR MISMATCH
    # ========================================================

    phi_obs_W = reconstructor.phi_west(
        x_obs,
        0.0,
        xE,
        lam,
    )

    phi_obs_E = reconstructor.phi_east(
        x_obs,
        Lx,
        xW,
        lam,
    )

    epsilon_p = (
        uW_obs
        -
        uE_obs
        -
        (
            a_W * phi_obs_W
            -
            a_E * phi_obs_E
        )
    )

    particular_mismatch = np.linalg.norm(
        epsilon_p
    )

    if generate_plots:

        plot_particular_solution_mismatch(
            x_obs=x_obs,
            epsilon_p=epsilon_p,
            run_dir=run_dir,
        )

    # ========================================================
    # SHARP RECONSTRUCTION
    # ========================================================

    u_sharp = np.zeros_like(
        u_full
    )

    mask_west_sharp = (
        x_full <= xI
    )

    mask_east_sharp = (
        x_full > xI
    )

    u_sharp[mask_west_sharp] = np.interp(
        x_full[mask_west_sharp],
        x_west,
        u_west_corr,
    )

    u_sharp[mask_east_sharp] = np.interp(
        x_full[mask_east_sharp],
        x_east,
        u_east_corr,
    )

    sharp_error = np.linalg.norm(
        u_sharp
        -
        u_full
    )

    # ========================================================
    # WEIGHTED RECONSTRUCTION
    # ========================================================

    u_weighted = np.zeros_like(
        u_full
    )

    mask_west = (
        x_full <= xW
    )

    mask_east = (
        x_full >= xE
    )

    mask_overlap = (
        (x_full > xW)
        &
        (x_full < xE)
    )

    u_weighted[mask_west] = np.interp(
        x_full[mask_west],
        x_west,
        u_west_corr,
    )

    u_weighted[mask_east] = np.interp(
        x_full[mask_east],
        x_east,
        u_east_corr,
    )

    if np.any(mask_overlap):

        uw = np.interp(
            x_full[mask_overlap],
            x_west,
            u_west_corr,
        )

        ue = np.interp(
            x_full[mask_overlap],
            x_east,
            u_east_corr,
        )

        omega_overlap = (
            reconstructor.linear_weight(
                x_full[mask_overlap],
                xW,
                xE,
            )
        )

        u_weighted[
            mask_overlap
        ] = (
            omega_overlap * uw
            +
            (1.0 - omega_overlap) * ue
        )

    weighted_error = np.linalg.norm(
        u_weighted
        -
        u_full
    )

    weighting_gain = (
        sharp_error
        /
        weighted_error
    )

    # ========================================================
    # UNCORRECTED ERROR
    # ========================================================

    u_uncorrected = np.zeros_like(
        u_full
    )

    u_uncorrected[mask_west] = np.interp(
        x_full[mask_west],
        x_west,
        u_west,
    )

    u_uncorrected[mask_east] = np.interp(
        x_full[mask_east],
        x_east,
        u_east,
    )

    if np.any(mask_overlap):

        uw = np.interp(
            x_full[mask_overlap],
            x_west,
            u_west,
        )

        ue = np.interp(
            x_full[mask_overlap],
            x_east,
            u_east,
        )

        omega_overlap = (
            reconstructor.linear_weight(
                x_full[mask_overlap],
                xW,
                xE,
            )
        )

        u_uncorrected[
            mask_overlap
        ] = (
            omega_overlap * uw
            +
            (1.0 - omega_overlap) * ue
        )

    # ========================================================
    # ERRORS
    # ========================================================

    l2_before = np.linalg.norm(
        u_uncorrected
        -
        u_full
    )

    l2_after = np.linalg.norm(
        u_weighted
        -
        u_full
    )

    west_before = np.linalg.norm(
        u_west - u_exact_west
    )

    west_after = np.linalg.norm(
        u_west_corr - u_exact_west
    )

    east_before = np.linalg.norm(
        u_east - u_exact_east
    )

    east_after = np.linalg.norm(
        u_east_corr - u_exact_east
    )

    west_improvement = (
        west_before
        /
        west_after
    )

    east_improvement = (
        east_before
        /
        east_after
    )

    global_improvement = (
        l2_before
        /
        l2_after
    )

    # ========================================================
    # FIGURES
    # ========================================================

    if generate_plots:

        plot_solution_with_error(
            x=x_full,
            u_exact=u_full,
            u_uncorrected=u_uncorrected,
            u_reconstructed=u_weighted,
            l2_before=l2_before,
            l2_after=l2_after,
            improvement=global_improvement,
            run_dir=run_dir,
        )

        plot_sharp_reconstruction(
            x=x_full,
            u_exact=u_full,
            u_sharp=u_sharp,
            sharp_error=sharp_error,
            run_dir=run_dir,
        )

        plot_weight_blending_comparison(
            x=x_full,
            u_exact=u_full,
            u_sharp=u_sharp,
            u_weighted=u_weighted,
            sharp_error=sharp_error,
            weighted_error=weighted_error,
            weighting_gain=weighting_gain,
            run_dir=run_dir,
        )

        plot_global_error_comparison(
            x=x_full,
            u_exact=u_full,
            u_uncorrected=u_uncorrected,
            u_sharp=u_sharp,
            u_weighted=u_weighted,
            run_dir=run_dir,
        )

        plot_overlap_detail(
            x_full=x_full,
            u_full=u_full,
            x_west=x_west,
            u_west=u_west,
            u_west_corr=u_west_corr,
            x_east=x_east,
            u_east=u_east,
            u_east_corr=u_east_corr,
            xW=xW,
            xE=xE,
            west_improvement=west_improvement,
            east_improvement=east_improvement,
            global_improvement=global_improvement,
            run_dir=run_dir,
        )

        plot_reconstruction_diagnostics(
            H=result.matrix,
            residual=result.residual,
            run_dir=run_dir,
        )

    return {
        "x": x_full,
        "u_exact": u_full,
        "u_uncorrected": u_uncorrected,
        "u_sharp": u_sharp,
        "u_weighted": u_weighted,
        "l2_before": l2_before,
        "l2_after": l2_after,
        "sharp_error": sharp_error,
        "weighted_error": weighted_error,
        "weighting_gain": weighting_gain,
        "condition_number": result.condition_number,
        "residual_norm": np.linalg.norm(result.residual),
        "particular_mismatch": particular_mismatch,
        "west_improvement": west_improvement,
        "east_improvement": east_improvement,
        "global_improvement": global_improvement,
    }


# ============================================================
# CONVERGENCE STUDY
# ============================================================

def run_convergence_study():
    """
    Mesh-refinement study.

    A single highly-resolved reference solution is used
    for all mesh levels.

    The reconstruction remains based on only two overlap
    observations:

        x_obs = [xW,xE]

    Returns
    -------
    pandas.DataFrame
    """

    logger.info("=" * 70)
    logger.info("Starting convergence study")
    logger.info("=" * 70)

    # ========================================================
    # REFERENCE SOLUTION
    # ========================================================

    n_reference = 10000

    logger.info(
        "Computing reference solution "
        "(n=%d)",
        n_reference,
    )

    x_ref, u_ref = solver.solve(
        a=0.0,
        b=Lx,
        n_elements=n_reference,
        bc_left=0.0,
        bc_right=0.0,
        current_time=current_time,
    )

    # ========================================================
    # TEST LEVELS
    # ========================================================

    mesh_sizes = [
        50,
        100,
        200,
        400,
        800,
        1600,
    ]

    physical_overlap_width = 2.0

    h_values = []

    error_before = []

    error_sharp = []

    error_weighted = []

    condition_numbers = []

    residual_norms = []

    mismatch_values = []

    overlap_widths = []

    # ========================================================
    # LOOP
    # ========================================================

    for n in mesh_sizes:

        logger.info(
            "Convergence level n=%d",
            n,
        )

        h = Lx / n

        # ----------------------------------------------------
        # keep overlap physically constant
        # ----------------------------------------------------

        delta_current = max(
            1,
            int(
                physical_overlap_width
                /
                h
            ),
        )

        overlap_width = (
            delta_current
            *
            h
        )

        overlap_widths.append(
            overlap_width
        )

        # ----------------------------------------------------
        # maintain approximately uniform
        # mesh density everywhere
        # ----------------------------------------------------

        local_xE = (
            xI
            +
            overlap_width / 2.0
        )

        local_xW = (
            xI
            -
            overlap_width / 2.0
        )

        n_west_local = max(
            10,
            int(
                n
                *
                local_xE
                /
                Lx
            ),
        )

        n_east_local = max(
            10,
            int(
                n
                *
                (
                    Lx
                    -
                    local_xW
                )
                /
                Lx
            ),
        )

        results = run_single_case(
            n_full=n,
            n_west=n_west_local,
            n_east=n_east_local,
            delta_west=delta_current,
            delta_east=delta_current,
            generate_plots=False,
        )

        # ----------------------------------------------------
        # reference interpolation
        # ----------------------------------------------------

        u_ref_interp = np.interp(
            results["x"],
            x_ref,
            u_ref,
        )

        # ----------------------------------------------------
        # proper discrete L2 norm
        # ----------------------------------------------------

        x = results["x"]

        dx = (
            x[-1]
            -
            x[0]
        ) / (
            len(x) - 1
        )

        before = np.sqrt(
            dx
            *
            np.sum(
                (
                    results[
                        "u_uncorrected"
                    ]
                    -
                    u_ref_interp
                )**2
            )
        )

        sharp = np.sqrt(
            dx
            *
            np.sum(
                (
                    results[
                        "u_sharp"
                    ]
                    -
                    u_ref_interp
                )**2
            )
        )

        weighted = np.sqrt(
            dx
            *
            np.sum(
                (
                    results[
                        "u_weighted"
                    ]
                    -
                    u_ref_interp
                )**2
            )
        )

        error_before.append(
            before
        )

        error_sharp.append(
            sharp
        )

        error_weighted.append(
            weighted
        )

        condition_numbers.append(
            results[
                "condition_number"
            ]
        )

        residual_norms.append(
            results[
                "residual_norm"
            ]
        )

        mismatch_values.append(
            results[
                "particular_mismatch"
            ]
        )

        h_values.append(
            h
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    h_values = np.asarray(
        h_values
    )

    error_before = np.asarray(
        error_before
    )

    error_sharp = np.asarray(
        error_sharp
    )

    error_weighted = np.asarray(
        error_weighted
    )

    # ========================================================
    # ORDERS
    # ========================================================

    fit_slice = slice(
        0,
        4,
    )

    order_before = abs(
        np.polyfit(
            np.log(
                h_values[fit_slice]
            ),
            np.log(
                error_before[
                    fit_slice
                ]
            ),
            1,
        )[0]
    )

    order_sharp = abs(
        np.polyfit(
            np.log(
                h_values[fit_slice]
            ),
            np.log(
                error_sharp[
                    fit_slice
                ]
            ),
            1,
        )[0]
    )

    order_weighted = abs(
        np.polyfit(
            np.log(
                h_values[fit_slice]
            ),
            np.log(
                error_weighted[
                    fit_slice
                ]
            ),
            1,
        )[0]
    )

    logger.info(
        "Order before   = %.3f",
        order_before,
    )

    logger.info(
        "Order sharp    = %.3f",
        order_sharp,
    )

    logger.info(
        "Order weighted = %.3f",
        order_weighted,
    )

    # ========================================================
    # PLOT
    # ========================================================

    plot_convergence(
        h=h_values,
        error_before=error_before,
        sharp_error=error_sharp,
        weighted_error=error_weighted,
        run_dir=run_dir,
    )

    # ========================================================
    # CONDITION NUMBER VS H
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.semilogy(
        h_values,
        condition_numbers,
        "o-",
        lw=2,
    )

    ax.grid(True)

    ax.set_xlabel(
        "Mesh size h"
    )

    ax.set_ylabel(
        "cond(H)"
    )

    ax.set_title(
        "Condition Number vs Mesh Size"
    )

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "condition_vs_h.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # PARTICULAR MISMATCH VS H
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.loglog(
        h_values,
        mismatch_values,
        "o-",
        lw=2,
    )

    ax.grid(True)

    ax.set_xlabel(
        "Mesh size h"
    )

    ax.set_ylabel(
        "Particular mismatch"
    )

    ax.set_title(
        "Particular Mismatch vs Mesh Size"
    )

    plt.tight_layout()

    plt.savefig(
        run_dir
        /
        "particular_mismatch_vs_h.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # DATAFRAME
    # ========================================================

    convergence_df = pd.DataFrame(
        {
            "mesh_size":
                mesh_sizes,

            "h":
                h_values,

            "overlap_width":
                overlap_widths,

            "error_before":
                error_before,

            "error_sharp":
                error_sharp,

            "error_weighted":
                error_weighted,

            "condition_number":
                condition_numbers,

            "particular_mismatch":
                mismatch_values,

            "residual_norm":
                residual_norms,
        }
    )

    convergence_df.to_csv(
        run_dir
        /
        "convergence.csv",
        index=False,
    )

    logger.info(
        "Finished convergence study."
    )

    return convergence_df

# ============================================================
# OVERLAP WIDTH STUDY
# ============================================================

def run_overlap_study():
    """
    Study reconstruction quality as a function
    of overlap width.

    The study investigates

        * condition number
        * reconstruction error
        * particular mismatch
        * weighting gain

    as the overlap thickness changes.

    Returns
    -------
    pandas.DataFrame
        Overlap-study statistics.
    """

    logger.info(
        "=" * 60
    )

    logger.info(
        "Starting overlap-width study."
    )

    logger.info(
        "=" * 60
    )

    overlap_cells = [
        1,
        2,
        3,
        5,
        10,
        15,
        20,
    ]

    overlap_widths = []

    condition_numbers = []

    reconstruction_errors = []

    particular_mismatches = []

    weighting_gains = []

    west_improvements = []

    east_improvements = []

    global_improvements = []

    residual_norms = []

    for overlap in overlap_cells:

        logger.info(
            "Testing overlap thickness = %d cells",
            overlap,
        )

        results = run_single_case(
            n_full=n_full,
            n_west=n_west,
            n_east=n_east,
            delta_west=overlap,
            delta_east=overlap,
            generate_plots=False
        )

        dx_local = (
            Lx
            /
            n_full
        )

        overlap_width = (
            2.0
            *
            overlap
            *
            dx_local
        )

        overlap_widths.append(
            overlap_width
        )

        condition_numbers.append(
            results[
                "condition_number"
            ]
        )

        reconstruction_errors.append(
            results[
                "l2_after"
            ]
        )

        particular_mismatches.append(
            results[
                "particular_mismatch"
            ]
        )

        weighting_gains.append(
            results[
                "weighting_gain"
            ]
        )

        west_improvements.append(
            results[
                "west_improvement"
            ]
        )

        east_improvements.append(
            results[
                "east_improvement"
            ]
        )

        global_improvements.append(
            results[
                "global_improvement"
            ]
        )

        residual_norms.append(
            results[
                "residual_norm"
            ]
        )

    # --------------------------------------------------------
    # ARRAYS
    # --------------------------------------------------------

    overlap_widths = np.asarray(
        overlap_widths
    )

    condition_numbers = np.asarray(
        condition_numbers
    )

    reconstruction_errors = np.asarray(
        reconstruction_errors
    )

    particular_mismatches = np.asarray(
        particular_mismatches
    )

    weighting_gains = np.asarray(
        weighting_gains
    )

    west_improvements = np.asarray(
        west_improvements
    )

    east_improvements = np.asarray(
        east_improvements
    )

    global_improvements = np.asarray(
        global_improvements
    )

    residual_norms = np.asarray(
        residual_norms
    )

    # --------------------------------------------------------
    # LOG SUMMARY
    # --------------------------------------------------------

    best_idx = np.argmin(
        reconstruction_errors
    )

    logger.info(
        "Minimum reconstruction error = %.6e",
        reconstruction_errors[
            best_idx
        ],
    )

    logger.info(
        "Best overlap width = %.6e",
        overlap_widths[
            best_idx
        ],
    )

    logger.info(
        "Maximum condition number = %.6e",
        np.max(
            condition_numbers
        ),
    )

    logger.info(
        "Minimum condition number = %.6e",
        np.min(
            condition_numbers
        ),
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_condition_vs_overlap(
        overlap_width=overlap_widths,
        condition_number=condition_numbers,
        run_dir=run_dir,
    )

    plot_error_vs_overlap(
        overlap_width=overlap_widths,
        reconstruction_error=reconstruction_errors,
        run_dir=run_dir,
    )

    plot_error_vs_condition(
        condition_number=condition_numbers,
        reconstruction_error=reconstruction_errors,
        run_dir=run_dir,
    )

    # --

# ============================================================
# EXPORT UTILITIES
# ============================================================

def export_single_case_results(
    results,
):
    """
    Export primary reconstruction results.

    Parameters
    ----------
    results : dict
        Dictionary returned by
        run_single_case().
    """

    df = pd.DataFrame(
        {
            "x":
                results["x"],

            "u_exact":
                results["u_exact"],

            "u_uncorrected":
                results["u_uncorrected"],

            "u_sharp":
                results["u_sharp"],

            "u_weighted":
                results["u_weighted"],
        }
    )

    df.to_csv(
        run_dir / "results.csv",
        index=False,
    )

    logger.info(
        "Saved results.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main validation driver.

    Workflow
    --------
    1. Execute a representative reconstruction case.
    2. Perform mesh-convergence study.
    3. Perform overlap-width study.
    4. Export CSV files.
    5. Report summary metrics.
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "Starting overlap reconstruction validation suite."
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # SINGLE CASE
    # --------------------------------------------------------

    logger.info(
        "Running primary validation case."
    )

    single_case_results = (
        run_single_case()
    )

    export_single_case_results(
        single_case_results
    )

    # --------------------------------------------------------
    # CONVERGENCE STUDY
    # --------------------------------------------------------

    logger.info(
        "Running convergence study."
    )

    convergence_results = (
        run_convergence_study()
    )

    convergence_results.to_csv(
        run_dir / "convergence.csv",
        index=False,
    )

    logger.info(
        "Saved convergence.csv"
    )

    # --------------------------------------------------------
    # OVERLAP STUDY
    # --------------------------------------------------------

    logger.info(
        "Running overlap-width study."
    )

    overlap_results = (
        run_overlap_study()
    )

    overlap_results.to_csv(
        run_dir / "overlap_study.csv",
        index=False,
    )

    logger.info(
        "Saved overlap_study.csv"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    logger.info(
        "=" * 70
    )

    logger.info(
        "Primary Case Summary"
    )

    logger.info(
        "-" * 70
    )

    logger.info(
        "Global L2 before      = %.6e",
        single_case_results[
            "l2_before"
        ],
    )

    logger.info(
        "Global L2 after       = %.6e",
        single_case_results[
            "l2_after"
        ],
    )

    logger.info(
        "Global improvement    = %.3f",
        single_case_results[
            "global_improvement"
        ],
    )

    logger.info(
        "West improvement      = %.3f",
        single_case_results[
            "west_improvement"
        ],
    )

    logger.info(
        "East improvement      = %.3f",
        single_case_results[
            "east_improvement"
        ],
    )

    logger.info(
        "Sharp error           = %.6e",
        single_case_results[
            "sharp_error"
        ],
    )

    logger.info(
        "Weighted error        = %.6e",
        single_case_results[
            "weighted_error"
        ],
    )

    logger.info(
        "Weighting gain        = %.3f",
        single_case_results[
            "weighting_gain"
        ],
    )

    logger.info(
        "Condition number      = %.6e",
        single_case_results[
            "condition_number"
        ],
    )

    logger.info(
        "Residual norm         = %.6e",
        single_case_results[
            "residual_norm"
        ],
    )

    logger.info(
        "Particular mismatch   = %.6e",
        single_case_results[
            "particular_mismatch"
        ],
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # CONVERGENCE SUMMARY
    # --------------------------------------------------------

    best_idx = (
        convergence_results[
            "weighted_error"
        ].idxmin()
    )

    logger.info(
        "Best convergence level"
    )

    logger.info(
        "h = %.6e",
        convergence_results.loc[
            best_idx,
            "h",
        ],
    )

    logger.info(
        "Weighted error = %.6e",
        convergence_results.loc[
            best_idx,
            "weighted_error",
        ],
    )

    # --------------------------------------------------------
    # OVERLAP SUMMARY
    # --------------------------------------------------------

    overlap_idx = (
        overlap_results[
            "reconstruction_error"
        ].idxmin()
    )

    logger.info(
        "Best overlap width = %.6e",
        overlap_results.loc[
            overlap_idx,
            "overlap_width",
        ],
    )

    logger.info(
        "Minimum overlap-study error = %.6e",
        overlap_results.loc[
            overlap_idx,
            "reconstruction_error",
        ],
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Validation suite completed."
    )

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Execute the complete validation framework.

    Workflow
    --------
    1. Run representative overlap reconstruction case.
    2. Run mesh-convergence study.
    3. Run overlap-width / conditioning study.
    4. Export CSV summaries.
    5. Report key diagnostics.
    """

    logger.info("=" * 70)
    logger.info("Starting overlap reconstruction validation suite.")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # PRIMARY VALIDATION CASE
    # --------------------------------------------------------

    logger.info(
        "Running primary reconstruction case."
    )

    primary = run_single_case()

    export_single_case_results(
        primary
    )

    # --------------------------------------------------------
    # CONVERGENCE STUDY
    # --------------------------------------------------------

    logger.info(
        "Running convergence study."
    )

    convergence_df = (
        run_convergence_study()
    )

    logger.info(
        "Saved convergence.csv"
    )

    # --------------------------------------------------------
    # OVERLAP STUDY
    # --------------------------------------------------------

    logger.info(
        "Running overlap-width study."
    )

    overlap_df = (
        run_overlap_study()
    )

    logger.info(
        "Saved overlap_study.csv"
    )

    # --------------------------------------------------------
    # PRIMARY SUMMARY
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Primary Case Summary")
    logger.info("=" * 70)

    logger.info(
        "Global L2 before       = %.6e",
        primary["l2_before"],
    )

    logger.info(
        "Global L2 after        = %.6e",
        primary["l2_after"],
    )

    logger.info(
        "Global improvement     = %.3f",
        primary["global_improvement"],
    )

    logger.info(
        "West improvement       = %.3f",
        primary["west_improvement"],
    )

    logger.info(
        "East improvement       = %.3f",
        primary["east_improvement"],
    )

    logger.info(
        "Sharp error            = %.6e",
        primary["sharp_error"],
    )

    logger.info(
        "Weighted error         = %.6e",
        primary["weighted_error"],
    )

    logger.info(
        "Weighting gain         = %.3f",
        primary["weighting_gain"],
    )

    logger.info(
        "Condition number       = %.6e",
        primary["condition_number"],
    )

    logger.info(
        "Residual norm          = %.6e",
        primary["residual_norm"],
    )

    logger.info(
        "Particular mismatch    = %.6e",
        primary["particular_mismatch"],
    )

    

    logger.info("=" * 70)
    logger.info("Validation suite completed.")
    logger.info("Output directory: %s", run_dir)
    logger.info("=" * 70)

    print()
    print("Validation suite completed.")
    print(f"Results stored in {run_dir}")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()