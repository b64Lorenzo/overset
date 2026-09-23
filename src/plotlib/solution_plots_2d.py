"""
solution_plots_2d.py
====================
"""

import matplotlib.pyplot as plt


def plot_surface(
    X,
    Y,
    U,
    title,
    filename,
):

    fig = plt.figure(
        figsize=(14, 10)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_trisurf(
        X,
        Y,
        U,
        cmap="viridis",
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=400,
    )

    plt.close()


def plot_contour(
    X,
    Y,
    U,
    title,
    filename,
):

    plt.figure(
        figsize=(12, 8)
    )

    plt.tricontourf(
        X,
        Y,
        U,
        levels=100,
        cmap="viridis",
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=400,
    )

    plt.close()


def plot_reference_solution(
    X,
    Y,
    U,
    output_dir,
):

    plot_surface(
        X,
        Y,
        U,
        "Reference Solution",
        output_dir
        /
        "reference_surface_3d.png",
    )

    plot_contour(
        X,
        Y,
        U,
        "Reference Solution",
        output_dir
        /
        "reference_contour_2d.png",
    )


def plot_reconstructed_solution(
    X,
    Y,
    U,
    output_dir,
):

    plot_surface(
        X,
        Y,
        U,
        "Reconstructed Overset Solution",
        output_dir
        /
        "reconstructed_surface_3d.png",
    )

    plot_contour(
        X,
        Y,
        U,
        "Reconstructed Overset Solution",
        output_dir
        /
        "reconstructed_contour_2d.png",
    )