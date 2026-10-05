"""
validation_plots_2d.py
======================
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_error_field(
    X,
    Y,
    error,
    title,
    filename,
):

    plt.figure(
        figsize=(12, 8)
    )

    plt.tricontourf(
        X,
        Y,
        error,
        levels=100,
        cmap="seismic",
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=400,
    )

    plt.close()


def plot_error_before(
    X,
    Y,
    error,
    output_dir,
):

    plot_error_field(
        X,
        Y,
        error,
        "Error Before Correction",
        output_dir
        /
        "error_before_2d.png",
    )


def plot_error_after(
    X,
    Y,
    error,
    output_dir,
):

    plot_error_field(
        X,
        Y,
        error,
        "Error After Correction",
        output_dir
        /
        "error_after_2d.png",
    )


def plot_modal_convergence(
    modes,
    errors,
    output_dir,
):

    plt.figure()

    plt.semilogy(
        modes,
        errors,
        "o-",
    )

    plt.xlabel(
        "Number of Modes"
    )

    plt.ylabel(
        "L2 Error"
    )

    plt.title(
        "Modal Convergence"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "error_vs_modes.png",
        dpi=400,
    )

    plt.close()


def plot_mesh_convergence(
    mesh_sizes,
    errors,
    output_dir,
):

    plt.figure()

    plt.loglog(
        mesh_sizes,
        errors,
        "o-",
    )

    plt.xlabel(
        "Mesh Size"
    )

    plt.ylabel(
        "L2 Error"
    )

    plt.title(
        "Mesh Convergence"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "mesh_convergence.png",
        dpi=400,
    )

    plt.close()


def plot_overlap_study(
    overlaps,
    errors,
    output_dir,
):

    plt.figure()

    plt.plot(
        overlaps,
        errors,
        "o-",
    )

    plt.xlabel(
        "Overlap Cells"
    )

    plt.ylabel(
        "L2 Error"
    )

    plt.title(
        "Overlap Width Study"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "error_vs_overlap.png",
        dpi=400,
    )

    plt.close()


def plot_observation_study(
    labels,
    errors,
    output_dir,
):

    plt.figure()

    plt.bar(
        labels,
        errors,
    )

    plt.ylabel(
        "L2 Error"
    )

    plt.title(
        "Observation Strategy Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "error_vs_sampling.png",
        dpi=400,
    )

    plt.close()


def plot_conditioning(
    labels,
    conds,
    output_dir,
):

    plt.figure()

    plt.bar(
        labels,
        conds,
    )

    plt.ylabel(
        "Condition Number"
    )

    plt.title(
        "System Conditioning"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "condition_vs_sampling.png",
        dpi=400,
    )

    plt.close()