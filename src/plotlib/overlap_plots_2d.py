"""
overlap_plots_2d.py
===================
"""

import matplotlib.pyplot as plt


def plot_observation_points(
    points,
    filename,
    title,
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=15,
        color="red",
    )

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=400,
    )

    plt.close()


def plot_two_line_observations(
    points,
    output_dir,
):

    plot_observation_points(
        points,
        output_dir
        /
        "two_line_observations.png",
        "Two-Line Observation Points",
    )


def plot_fps_observations(
    points,
    output_dir,
):

    plot_observation_points(
        points,
        output_dir
        /
        "fps_observations.png",
        "FPS Observation Points",
    )


def plot_overlap_geometry(
    west_points,
    east_points,
    output_dir,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        west_points[:, 0],
        west_points[:, 1],
        s=2,
        label="West Domain",
    )

    plt.scatter(
        east_points[:, 0],
        east_points[:, 1],
        s=2,
        label="East Domain",
    )

    plt.legend()

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title(
        "Overlap Geometry"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "overlap_geometry.png",
        dpi=400,
    )

    plt.close()