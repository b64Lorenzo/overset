"""
validation_plots.py
===================

Author
------
Lorenzo Zambelli

Version
-------
1.0

Description
-----------
Validation and diagnostic visualization utilities for the
one-dimensional overlap reconstruction framework.

The purpose of this module is to assess

* reconstruction-system quality,
* residual behavior,
* convergence properties,
* condition-number effects,
* overlap-width sensitivity,
* reconstruction robustness.

Generated Figures
-----------------
reconstruction_diagnostics.png
    Matrix H and reconstruction residual.

convergence_h.png
    Error convergence under mesh refinement.

condition_vs_overlap.png
    Condition number as a function of overlap width.

error_vs_overlap.png
    Reconstruction error as a function of overlap width.

error_vs_condition.png
    Reconstruction error versus condition number.

Dependencies
------------
numpy
matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# RECONSTRUCTION DIAGNOSTICS
# ============================================================

def plot_reconstruction_diagnostics(
    H,
    residual,
    run_dir,
):
    """
    Plot reconstruction matrix and residual.

    Parameters
    ----------
    H : ndarray
        Reconstruction matrix.

    residual : ndarray
        Residual vector

            r = d - Ha.

    run_dir : Path
        Output directory.
    """

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # --------------------------------------------------------
    # MATRIX
    # --------------------------------------------------------

    image = ax[0].imshow(
        H,
        aspect="auto",
        cmap="viridis",
    )

    ax[0].set_title(
        "Reconstruction Matrix H"
    )

    ax[0].set_xlabel(
        "Column Index"
    )

    ax[0].set_ylabel(
        "Row Index"
    )

    plt.colorbar(
        image,
        ax=ax[0],
    )

    # --------------------------------------------------------
    # RESIDUAL
    # --------------------------------------------------------

    residual_norm = np.linalg.norm(
        residual
    )

    ax[1].plot(
        residual,
        "o-",
        linewidth=2,
    )

    ax[1].grid(True)

    ax[1].set_title(
        "Reconstruction Residual"
    )

    ax[1].set_xlabel(
        "Observation Index"
    )

    ax[1].set_ylabel(
        "Residual"
    )

    ax[1].text(
        0.02,
        0.95,
        f"||r||₂ = {residual_norm:.3e}",
        transform=ax[1].transAxes,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
        ),
    )

    plt.tight_layout()

    plt.savefig(
        Path(run_dir)
        /
        "reconstruction_diagnostics.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# CONVERGENCE STUDY
# ============================================================

def plot_convergence(
    h,
    error_before,
    sharp_error,
    weighted_error,
    run_dir,
):
    """
    Plot convergence under mesh refinement.

    Parameters
    ----------
    h : ndarray
        Characteristic mesh size.

    error_before : ndarray
        Error before reconstruction.

    sharp_error : ndarray
        Error after sharp assembly.

    weighted_error : ndarray
        Error after weighted assembly.

    run_dir : Path
        Output directory.
    """

    h = np.asarray(h)

    error_before = np.asarray(
        error_before
    )

    sharp_error = np.asarray(
        sharp_error
    )

    weighted_error = np.asarray(
        weighted_error
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.loglog(
        h,
        error_before,
        "o-",
        linewidth=2,
        label="Before reconstruction",
    )

    ax.loglog(
        h,
        sharp_error,
        "s-",
        linewidth=2,
        label="Sharp reconstruction",
    )

    ax.loglog(
        h,
        weighted_error,
        "^-",
        linewidth=2,
        label="Weighted reconstruction",
    )

    # --------------------------------------------------------
    # Estimated Orders
    # --------------------------------------------------------

    p_before = np.polyfit(
        np.log(h),
        np.log(error_before),
        1,
    )[0]

    p_sharp = np.polyfit(
        np.log(h),
        np.log(sharp_error),
        1,
    )[0]

    p_weighted = np.polyfit(
        np.log(h),
        np.log(weighted_error),
        1,
    )[0]

    txt = (
        f"Order before   = {abs(p_before):.2f}\n"
        f"Order sharp    = {abs(p_sharp):.2f}\n"
        f"Order weighted = {abs(p_weighted):.2f}"
    )

    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
        ),
    )

    ax.grid(
        which="both",
        linestyle="--",
    )

    ax.set_xlabel(
        "Mesh size h"
    )

    ax.set_ylabel(
        "L2 Error"
    )

    ax.legend()

    ax.set_title(
        "Convergence Study"
    )

    plt.tight_layout()

    plt.savefig(
        Path(run_dir)
        /
        "convergence_h.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# CONDITION NUMBER VS OVERLAP
# ============================================================

def plot_condition_vs_overlap(
    overlap_width,
    condition_number,
    run_dir,
):
    """
    Plot condition number versus overlap width.

    Parameters
    ----------
    overlap_width : ndarray
        Physical overlap width.

    condition_number : ndarray
        cond(H).

    run_dir : Path
        Output directory.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.semilogy(
        overlap_width,
        condition_number,
        "o-",
        linewidth=2,
    )

    ax.grid(
        which="both"
    )

    ax.set_xlabel(
        "Overlap Width"
    )

    ax.set_ylabel(
        "Condition Number"
    )

    ax.set_title(
        "Condition Number vs Overlap Width"
    )

    plt.tight_layout()

    plt.savefig(
        Path(run_dir)
        /
        "condition_vs_overlap.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ERROR VS OVERLAP
# ============================================================

def plot_error_vs_overlap(
    overlap_width,
    reconstruction_error,
    run_dir,
):
    """
    Plot reconstruction error as a function
    of overlap width.

    Parameters
    ----------
    overlap_width : ndarray
        Overlap width.

    reconstruction_error : ndarray
        Reconstruction error.

    run_dir : Path
        Output directory.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        overlap_width,
        reconstruction_error,
        "o-",
        linewidth=2,
    )

    ax.grid(True)

    ax.set_xlabel(
        "Overlap Width"
    )

    ax.set_ylabel(
        "L2 Reconstruction Error"
    )

    ax.set_title(
        "Error vs Overlap Width"
    )

    plt.tight_layout()

    plt.savefig(
        Path(run_dir)
        /
        "error_vs_overlap.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ERROR VS CONDITION NUMBER
# ============================================================

def plot_error_vs_condition(
    condition_number,
    reconstruction_error,
    run_dir,
):
    """
    Plot reconstruction error versus
    reconstruction-system condition number.

    Parameters
    ----------
    condition_number : ndarray
        cond(H).

    reconstruction_error : ndarray
        Global reconstruction error.

    run_dir : Path
        Output directory.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.loglog(
        condition_number,
        reconstruction_error,
        "o-",
        linewidth=2,
    )

    ax.grid(
        which="both"
    )

    ax.set_xlabel(
        "Condition Number"
    )

    ax.set_ylabel(
        "L2 Reconstruction Error"
    )

    ax.set_title(
        "Error vs Condition Number"
    )

    plt.tight_layout()

    plt.savefig(
        Path(run_dir)
        /
        "error_vs_condition.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)