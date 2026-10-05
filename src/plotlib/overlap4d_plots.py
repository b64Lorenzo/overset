"""
overlap4d_plots.py
==================

Plotting utilities for four-domain overlap reconstruction studies.

Author
------
Lorenzo Zambelli
"""

from __future__ import annotations

import numpy as np

from scipy.interpolate import LinearNDInterpolator

from src.plotlib.plot import (
    plot_field,
    plot_subdomains,
)


# ============================================================
# SUBDOMAIN PLOTS
# ============================================================

def plot_overlap4d_subdomains(
    subdomains,
    run_dir,
):
    """
    Plot all subdomain quantities.

    Inputs:
        subdomains : list[dict]
        run_dir : pathlib.Path
    """

    plot_subdomains(
        subdomains,
        run_dir,
    )


# ============================================================
# REFERENCE SOLUTION
# ============================================================

def plot_reference_solution(
    X,
    Y,
    u,
    run_dir,
):
    """
    Plot reference solution.

    Inputs:
        X,Y : ndarray
        u : ndarray
        run_dir
    """

    plot_field(
        X,
        Y,
        u,
        "Reference Solution",
        "reference_2d.png",
        run_dir,
    )


# ============================================================
# RECONSTRUCTED SOLUTION
# ============================================================

def plot_reconstructed_solution(
    X,
    Y,
    u,
    run_dir,
):
    """
    Plot reconstructed solution.

    Inputs:
        X,Y : ndarray
        u : ndarray
        run_dir
    """

    plot_field(
        X,
        Y,
        u,
        "Reconstructed Solution",
        "reconstructed_2d.png",
        run_dir,
    )


# ============================================================
# ERROR BEFORE
# ============================================================

def plot_error_before(
    X,
    Y,
    error,
    run_dir,
):
    """
    Plot reconstruction error before correction.

    Inputs:
        X,Y : ndarray
        error : ndarray
        run_dir
    """

    plot_field(
        X,
        Y,
        error,
        "Error Before Correction",
        "error_before_2d.png",
        run_dir,
        cmap="seismic",
    )


# ============================================================
# ERROR AFTER
# ============================================================

def plot_error_after(
    X,
    Y,
    error,
    run_dir,
):
    """
    Plot reconstruction error after correction.

    Inputs:
        X,Y : ndarray
        error : ndarray
        run_dir
    """

    plot_field(
        X,
        Y,
        error,
        "Error After Correction",
        "error_after_2d.png",
        run_dir,
        cmap="seismic",
    )


# ============================================================
# GLOBAL FIELDS
# ============================================================

def build_uncorrected_global_field(
    basis_list,
    solution_list,
    Xref,
    Yref,
):
    """
    Construct global field by averaging raw subdomain solutions.

    Inputs:
        basis_list : list
        solution_list : list
        Xref,Yref : ndarray

    Outputs:
        u_unc : ndarray
    """

    interpolators = [

        LinearNDInterpolator(
            np.c_[
                basis.doflocs[0],
                basis.doflocs[1],
            ],
            solution,
            fill_value=np.nan,
        )

        for basis, solution
        in zip(
            basis_list,
            solution_list,
        )
    ]

    u_unc = np.zeros_like(Xref)

    for i, (x, y) in enumerate(
        zip(Xref, Yref)
    ):

        vals = []

        for interp in interpolators:

            value = interp(
                x,
                y,
            )

            if not np.isnan(value):

                vals.append(
                    float(value)
                )

        u_unc[i] = (
            np.mean(vals)
            if len(vals) > 0
            else np.nan
        )

    return u_unc


# ============================================================
# RECONSTRUCTED GLOBAL FIELD
# ============================================================

def build_reconstructed_global_field(
    result,
    Xref,
    Yref,
):
    """
    Evaluate reconstructed global solution.

    Inputs:
        result : OversetCorrectionResult2D
        Xref,Yref : ndarray

    Outputs:
        u_rec : ndarray
    """

    interp = LinearNDInterpolator(
        np.c_[
            result.full_x,
            result.full_y,
        ],
        result.full_solution,
        fill_value=np.nan,
    )

    return interp(
        Xref,
        Yref,
    )


# ============================================================
# ERRORS
# ============================================================

def compute_errors(
    u_ref,
    u_unc,
    u_rec,
):
    """
    Compute pre- and post-correction errors.

    Inputs:
        u_ref : ndarray
        u_unc : ndarray
        u_rec : ndarray

    Outputs:
        dict
    """

    err_before = np.zeros_like(u_ref)
    err_after = np.zeros_like(u_ref)

    mask_before = (
        ~np.isnan(u_ref)
        &
        ~np.isnan(u_unc)
    )

    mask_after = (
        ~np.isnan(u_ref)
        &
        ~np.isnan(u_rec)
    )

    err_before[mask_before] = (
        u_unc[mask_before]
        -
        u_ref[mask_before]
    )

    err_after[mask_after] = (
        u_rec[mask_after]
        -
        u_ref[mask_after]
    )

    return {

        "err_before":
        err_before,

        "err_after":
        err_after,

        "L2_before":
        np.linalg.norm(
            err_before[mask_before]
        ),

        "L2_after":
        np.linalg.norm(
            err_after[mask_after]
        ),

        "mask_before":
        mask_before,

        "mask_after":
        mask_after,
    }


# ============================================================
# COMPLETE POSTPROCESSING
# ============================================================

def postprocess_overlap4d(
    basis_ref,
    u_ref,
    domains,
    result,
):
    """
    Evaluate reconstructed and uncorrected errors.

    Inputs:
        basis_ref
        u_ref
        domains
        result

    Outputs:
        dict
    """

    Xref = basis_ref.doflocs[0]
    Yref = basis_ref.doflocs[1]

    basis_list = [
        d_basis
        for d_basis
        in [
            d["basis"]
            for d in domains
        ]
    ]

    solution_list = [
        d["u"]
        for d in domains
    ]

    u_unc = build_uncorrected_global_field(
        basis_list,
        solution_list,
        Xref,
        Yref,
    )

    u_rec = build_reconstructed_global_field(
        result,
        Xref,
        Yref,
    )

    out = compute_errors(
        u_ref,
        u_unc,
        u_rec,
    )

    out["u_unc"] = u_unc
    out["u_rec"] = u_rec
    out["X"] = Xref
    out["Y"] = Yref

    return out
