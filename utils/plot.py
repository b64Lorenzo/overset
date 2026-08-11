#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator


from matplotlib.colors import TwoSlopeNorm


def plot_field(
    X,
    Y,
    U,
    title,
    filename,
    run_dir,
    cmap="viridis",
):

    plt.figure(figsize=(10, 8))

    norm = None

    # center diverging colormaps at zero
    if cmap in ["seismic", "bwr", "coolwarm", "RdBu", "RdBu_r"]:

        vmax = np.max(
            np.abs(U)
        )

        norm = TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0.0,
            vmax=vmax,
        )

    plt.tricontourf(
        X,
        Y,
        U,
        levels=100,
        cmap=cmap,
        norm=norm,
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        run_dir / filename,
        dpi=400,
    )

    plt.close()



def plot_surface(
    X,
    Y,
    U,
    title,
    filename,
    run_dir,
    cmap="viridis",
):

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_trisurf(
        X,
        Y,
        U,
        cmap=cmap,
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        run_dir / filename,
        dpi=400,
    )

    plt.close()


from scipy.interpolate import LinearNDInterpolator
import numpy as np


from scipy.interpolate import LinearNDInterpolator
import numpy as np


def build_global_interpolator(
    subdomains,
    key,
    geometry="cross",
    assembly="sharp",
    xI=None,
    yI=None,
    xI_left=None,
    xI_right=None,
    yI_top=None,
    yI_bottom=None,
):
    """
    Build a global interpolator from overset subdomains.

    Parameters
    ----------
    subdomains : list[dict]

        Vertical split:
            names = ["A", "B"]

        Horizontal split:
            names = ["A", "B"]

        Cross split:
            names = ["D1", "D2", "D3", "D4"]

        Each entry must contain:

            {
                "name": str,
                "basis": basis,
                key: ndarray,
            }

    key : str

        Field to assemble,
        e.g.

            "u"
            "uc"

    geometry : str

        One of

            "vertical"
            "horizontal"
            "cross"

    assembly : str

        "sharp"

            No overlap in final reconstruction.

        "blend"

            Overset blending retained in final reconstruction.

    xI, yI

        Physical interfaces.

    xI_left, xI_right

        Vertical overlap boundaries.

    yI_top, yI_bottom

        Horizontal overlap boundaries.

    Returns
    -------
    callable

        interp_global(x,y)
    """

    # ======================================================
    # PARAMETER CHECKING
    # ======================================================

    if geometry not in (
        "vertical",
        "horizontal",
        "cross",
    ):
        raise ValueError(
            f"Unknown geometry '{geometry}'"
        )

    if assembly not in (
        "sharp",
        "blend",
    ):
        raise ValueError(
            f"Unknown assembly '{assembly}'"
        )

    if geometry == "vertical":

        names = {
            sd["name"]
            for sd in subdomains
        }

        if names != {"A", "B"}:
            raise ValueError(
                "Vertical geometry requires "
                "subdomains named A and B."
            )

        if assembly == "sharp":

            if xI is None:
                raise ValueError(
                    "Vertical sharp assembly "
                    "requires xI."
                )

        else:

            if (
                xI_left is None
                or
                xI_right is None
            ):
                raise ValueError(
                    "Vertical blend assembly "
                    "requires xI_left and "
                    "xI_right."
                )

    if geometry == "horizontal":

        names = {
            sd["name"]
            for sd in subdomains
        }

        if names != {"A", "B"}:
            raise ValueError(
                "Horizontal geometry requires "
                "subdomains named A and B."
            )

        if assembly == "sharp":

            if yI is None:
                raise ValueError(
                    "Horizontal sharp assembly "
                    "requires yI."
                )

        else:

            if (
                yI_top is None
                or
                yI_bottom is None
            ):
                raise ValueError(
                    "Horizontal blend assembly "
                    "requires yI_top and "
                    "yI_bottom."
                )

    if geometry == "cross":

        names = {
            sd["name"]
            for sd in subdomains
        }

        if names != {
            "D1",
            "D2",
            "D3",
            "D4",
        }:
            raise ValueError(
                "Cross geometry requires "
                "D1, D2, D3 and D4."
            )

        if assembly == "sharp":

            if xI is None or yI is None:
                raise ValueError(
                    "Cross sharp assembly "
                    "requires xI and yI."
                )

        else:

            if (
                xI_left is None
                or xI_right is None
                or yI_top is None
                or yI_bottom is None
            ):
                raise ValueError(
                    "Cross blend assembly "
                    "requires xI_left, "
                    "xI_right, "
                    "yI_top and "
                    "yI_bottom."
                )

    # ======================================================
    # LOCAL INTERPOLATORS
    # ======================================================

    lookup = {}

    for sd in subdomains:

        lookup[sd["name"]] = (
            LinearNDInterpolator(
                np.c_[
                    sd["basis"].doflocs[0],
                    sd["basis"].doflocs[1],
                ],
                sd[key],
            )
        )

    # ======================================================
    # GLOBAL INTERPOLATOR
    # ======================================================

    def interp_global(x, y):

        x = np.asarray(x)
        y = np.asarray(y)

        u = np.empty_like(
            x,
            dtype=float,
        )

        # ==================================================
        # VERTICAL
        # ==================================================

        if geometry == "vertical":

            A = lookup["A"]
            B = lookup["B"]

            # ----------------------------------------------
            # SHARP
            # ----------------------------------------------

            if assembly == "sharp":

                maskA = x <= xI
                maskB = x > xI

                if np.any(maskA):

                    u[maskA] = A(
                        x[maskA],
                        y[maskA],
                    )

                if np.any(maskB):

                    u[maskB] = B(
                        x[maskB],
                        y[maskB],
                    )

            # ----------------------------------------------
            # BLEND
            # ----------------------------------------------

            else:

                maskA = x < xI_right

                maskB = x > xI_left

                maskO = (
                    (x >= xI_right)
                    &
                    (x <= xI_left)
                )

                if np.any(maskA):

                    u[maskA] = A(
                        x[maskA],
                        y[maskA],
                    )

                if np.any(maskB):

                    u[maskB] = B(
                        x[maskB],
                        y[maskB],
                    )

                if np.any(maskO):

                    wA = (
                        xI_left
                        - x[maskO]
                    ) / (
                        xI_left
                        - xI_right
                    )

                    wB = 1.0 - wA

                    u[maskO] = (
                        wA * A(
                            x[maskO],
                            y[maskO],
                        )
                        +
                        wB * B(
                            x[maskO],
                            y[maskO],
                        )
                    )

        # ==================================================
        # HORIZONTAL
        # ==================================================

        elif geometry == "horizontal":

            A = lookup["A"]
            B = lookup["B"]

            if assembly == "sharp":

                maskA = y <= yI
                maskB = y > yI

                if np.any(maskA):
                    u[maskA] = A(
                        x[maskA],
                        y[maskA],
                    )

                if np.any(maskB):
                    u[maskB] = B(
                        x[maskB],
                        y[maskB],
                    )

            else:

                maskA = y < yI_top

                maskB = y > yI_bottom

                maskO = (
                    (y >= yI_top)
                    &
                    (y <= yI_bottom)
                )

                if np.any(maskA):
                    u[maskA] = A(
                        x[maskA],
                        y[maskA],
                    )

                if np.any(maskB):
                    u[maskB] = B(
                        x[maskB],
                        y[maskB],
                    )

                if np.any(maskO):

                    wA = (
                        yI_bottom
                        - y[maskO]
                    ) / (
                        yI_bottom
                        - yI_top
                    )

                    wB = 1.0 - wA

                    u[maskO] = (
                        wA * A(
                            x[maskO],
                            y[maskO],
                        )
                        +
                        wB * B(
                            x[maskO],
                            y[maskO],
                        )
                    )

        # ==================================================
        # CROSS
        # ==================================================

        elif geometry == "cross":

            D1 = lookup["D1"]
            D2 = lookup["D2"]
            D3 = lookup["D3"]
            D4 = lookup["D4"]

            if assembly == "sharp":

                mask_D1 = (
                    (x <= xI)
                    &
                    (y <= yI)
                )

                mask_D2 = (
                    (x > xI)
                    &
                    (y <= yI)
                )

                mask_D3 = (
                    (x <= xI)
                    &
                    (y > yI)
                )

                mask_D4 = (
                    (x > xI)
                    &
                    (y > yI)
                )

                if np.any(mask_D1):
                    u[mask_D1] = D1(
                        x[mask_D1],
                        y[mask_D1],
                    )

                if np.any(mask_D2):
                    u[mask_D2] = D2(
                        x[mask_D2],
                        y[mask_D2],
                    )

                if np.any(mask_D3):
                    u[mask_D3] = D3(
                        x[mask_D3],
                        y[mask_D3],
                    )

                if np.any(mask_D4):
                    u[mask_D4] = D4(
                        x[mask_D4],
                        y[mask_D4],
                    )

            else:

                wx = np.clip(
                    (xI_left - x)
                    /
                    (xI_left - xI_right),
                    0.0,
                    1.0,
                )

                wy = np.clip(
                    (yI_bottom - y)
                    /
                    (yI_bottom - yI_top),
                    0.0,
                    1.0,
                )

                w1 = wx * wy
                w2 = (1.0 - wx) * wy
                w3 = wx * (1.0 - wy)
                w4 = (1.0 - wx) * (1.0 - wy)

                v1 = D1(x, y)
                v2 = D2(x, y)
                v3 = D3(x, y)
                v4 = D4(x, y)

                u[:] = 0.0

                weight_sum = np.zeros_like(
                    x,
                    dtype=float,
                )

                m1 = np.isfinite(v1)
                m2 = np.isfinite(v2)
                m3 = np.isfinite(v3)
                m4 = np.isfinite(v4)

                if np.any(m1):
                    u[m1] += w1[m1] * v1[m1]
                    weight_sum[m1] += w1[m1]

                if np.any(m2):
                    u[m2] += w2[m2] * v2[m2]
                    weight_sum[m2] += w2[m2]

                if np.any(m3):
                    u[m3] += w3[m3] * v3[m3]
                    weight_sum[m3] += w3[m3]

                if np.any(m4):
                    u[m4] += w4[m4] * v4[m4]
                    weight_sum[m4] += w4[m4]

                valid = weight_sum > 0.0

                u[valid] /= weight_sum[valid]

                u[~valid] = np.nan

        return u

    return interp_global
    
def plot_subdomains(
    subdomains,
    run_dir,
):

    for sd in subdomains:

        X,Y = sd["basis"].doflocs

        plot_field(
            X,
            Y,
            sd["u"],
            f"{sd['name']} original",
            f"{sd['name']}_original.png",
            run_dir,
        )

        plot_field(
            X,
            Y,
            sd["uc"],
            f"{sd['name']} corrected",
            f"{sd['name']}_corrected.png",
            run_dir,
        )

        plot_field(
            X,
            Y,
            sd["correction"],
            f"{sd['name']} correction",
            f"{sd['name']}_correction.png",
            run_dir,
        )


def export_subdomains(
    subdomains,
    run_dir,
):

    for sd in subdomains:

        X,Y = sd["basis"].doflocs

        pd.DataFrame(
            {
                "x": X,
                "y": Y,
                "u_original": sd["u"],
                "u_corrected": sd["uc"],
                "correction": sd["correction"],
            }
        ).to_csv(
            run_dir /
            f"{sd['name']}.csv",
            index=False,
        )