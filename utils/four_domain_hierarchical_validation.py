#!/usr/bin/env python3
"""
four_domain_hierarchical_validation.py

Hierarchical reconstruction workflow:

    D3 | D4
    -------
    D1 | D2

Step 1:
    D1 <-> D2
    D3 <-> D4

Step 2:
    (D1+D2) <-> (D3+D4)

This file contains the reconstruction orchestration logic and assumes
InterfaceCorrector, InterfaceGeometry, solve_rect and Logger are available.
"""

#!/usr/bin/env python3

import numpy as np

from utils.interface_correction import (
    InterfaceGeometry,
    InterfaceCorrector,
)

def make_geometry(
    interface_type,
    location,
    alphas,
    omegas,
):
    """
    Supports both

        alpha = scalar
        omega = scalar

    and

        alphas = array
        omegas = array
    """

    if np.ndim(omegas) == 0:

        return InterfaceGeometry(
            interface_type=interface_type,
            location=location,
            decay_rate=float(alphas),
            omega=float(omegas),
        )

    return InterfaceGeometry(
        interface_type=interface_type,
        location=location,
        decay_rates=np.asarray(alphas),
        omegas=np.asarray(omegas),
    )


def hierarchical_reconstruct(

    basis1, u1,
    basis2, u2,
    basis3, u3,
    basis4, u4,

    interp_full,

    xmid,
    ymid,

    alpha_vertical,
    alpha_horizontal,

    omega_vertical,
    omega_horizontal,

    logger=None,
):

    # ======================================================
    # LEVEL 1
    # D1 <-> D2
    # ======================================================

    if logger:
        logger.info(
            "LEVEL-1 : D1 <-> D2"
        )

    res12 = InterfaceCorrector.correct(

        basisA=basis1,
        uA=u1,

        basisB=basis2,
        uB=u2,

        interp_reference=interp_full,

        geometry=make_geometry(
            interface_type="vertical",
            location=xmid,
            alphas=alpha_vertical,
            omegas=omega_vertical,
        ),

        logger=logger,
    )

    # ======================================================
    # LEVEL 1
    # D3 <-> D4
    # ======================================================

    if logger:
        logger.info(
            "LEVEL-1 : D3 <-> D4"
        )

    res34 = InterfaceCorrector.correct(

        basisA=basis3,
        uA=u3,

        basisB=basis4,
        uB=u4,

        interp_reference=interp_full,

        geometry=make_geometry(
            interface_type="vertical",
            location=xmid,
            alphas=alpha_vertical,
            omegas=omega_vertical,
        ),

        logger=logger,
    )

    # ======================================================
    # APPLY LEVEL-1 CORRECTIONS
    # ======================================================

    u1c = res12.corrected_A
    u2c = res12.corrected_B

    u3c = res34.corrected_A
    u4c = res34.corrected_B

    # ======================================================
    # LEVEL 2
    # D1 <-> D3
    # ======================================================

    if logger:
        logger.info(
            "LEVEL-2 : D1 <-> D3"
        )

    res13 = InterfaceCorrector.correct(

        basisA=basis1,
        uA=u1c,

        basisB=basis3,
        uB=u3c,

        interp_reference=interp_full,

        geometry=make_geometry(
            interface_type="horizontal",
            location=ymid,
            alphas=alpha_horizontal,
            omegas=omega_horizontal,
        ),

        logger=logger,
    )

    # ======================================================
    # LEVEL 2
    # D2 <-> D4
    # ======================================================

    if logger:
        logger.info(
            "LEVEL-2 : D2 <-> D4"
        )

    res24 = InterfaceCorrector.correct(

        basisA=basis2,
        uA=u2c,

        basisB=basis4,
        uB=u4c,

        interp_reference=interp_full,

        geometry=make_geometry(
            interface_type="horizontal",
            location=ymid,
            alphas=alpha_horizontal,
            omegas=omega_horizontal,
        ),

        logger=logger,
    )

    # ======================================================
    # ACCUMULATE ALL CORRECTIONS
    # ======================================================

    corr1 = (
        res12.correction_A
        + res13.correction_A
    )

    corr2 = (
        res12.correction_B
        + res24.correction_A
    )

    corr3 = (
        res34.correction_A
        + res13.correction_B
    )

    corr4 = (
        res34.correction_B
        + res24.correction_B
    )

    # ======================================================
    # FINAL SOLUTIONS
    # ======================================================

    u1f = u1 - corr1
    u2f = u2 - corr2

    u3f = u3 - corr3
    u4f = u4 - corr4

    # ======================================================
    # RETURN
    # ======================================================

    return (

        u1f,
        u2f,
        u3f,
        u4f,

        res12,
        res34,

        res13,
        res24,

        corr1,
        corr2,
        corr3,
        corr4,
    )