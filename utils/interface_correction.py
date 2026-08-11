#!/usr/bin/env python3
"""
interface_correction.py
=======================

Generic interface reconstruction framework for finite-element
domain-decomposition methods.

Overview
--------
This module implements the homogeneous reconstruction methodology
developed during the two-domain validation studies.

The objective is to identify, approximate, and remove the dominant
homogeneous error component introduced by imperfect interface
boundary data.

The implementation is intentionally independent of the particular
Schwarz strategy being used. Consequently, it may be reused for

    * Two-subdomain decompositions
    * Four-subdomain decompositions
    * Hierarchical domain decomposition
    * Recursive correction procedures
    * Validation studies using reference solutions

The framework assumes two neighbouring finite-element subdomains
sharing exactly one interface.

Supported interfaces
--------------------

Vertical interface

    x = constant

Horizontal interface

    y = constant

Mathematical Formulation
------------------------

Vertical Interface
~~~~~~~~~~~~~~~~~~

For

        Domain A | Domain B

with

        x = xI

the homogeneous error is approximated by

    eA(x,y)
    =
    AA exp(-alpha (xI - x))
    sin(omega_y y)

    eB(x,y)
    =
    AB exp(-alpha (x - xI))
    sin(omega_y y)

The unknown amplitudes

    AA
    AB

are obtained by solving a least-squares system built from
interior matching planes.

Horizontal Interface
~~~~~~~~~~~~~~~~~~~~

For

            Domain B
        ----------------
            Domain A

with

        y = yI

the homogeneous error becomes

    eA(x,y)
    =
    AA exp(-beta (yI - y))
    sin(omega_x x)

    eB(x,y)
    =
    AB exp(-beta (y - yI))
    sin(omega_x x)

Again, the amplitudes are determined through least-squares fitting.

Logging
-------
The module is fully compatible with

    utils.logger.Logger

used throughout the validation framework.

If a logger is supplied, detailed information is emitted through

    logger.info(...)
    logger.debug(...)

covering:

    * Interface geometry
    * Matching planes
    * Least-squares system size
    * Reconstructed coefficients
    * Error reduction statistics

Typical Usage
-------------

>>> geometry = InterfaceGeometry(
...     interface_type="vertical",
...     location=xI,
...     decay_rate=alpha,
...     omega=omega_y,
... )

>>> result = InterfaceCorrector.correct(
...     basisA=basisL,
...     uA=uL,
...     basisB=basisR,
...     uB=uR,
...     interp_reference=interp_full,
...     geometry=geometry,
...     logger=logger,
... )

>>> uLc = result.corrected_A
>>> uRc = result.corrected_B

>>> eL = result.correction_A
>>> eR = result.correction_B

>>> AL = result.coefficient_A
>>> AR = result.coefficient_B

Notes
-----
The implementation assumes:

    * One shared interface
    * Continuous P1 finite elements
    * A reference interpolator is available

"""

from dataclasses import dataclass
import numpy as np


# ==========================================================
# INTERFACE DESCRIPTION
# ==========================================================

@dataclass
class InterfaceGeometry:
    """
    Description of a shared interface between two neighbouring
    finite-element subdomains.

    This class stores the geometric and modal information required
    by the reconstruction procedure.

    Parameters
    ----------
    interface_type : str

        Type of interface.

        Allowed values

            "vertical"
            "horizontal"

    location : float

        Physical location of the interface.

        Vertical interface

            x = location

        Horizontal interface

            y = location

    decay_rate : float

        Exponential decay rate.

        Typical values

            alpha
            beta

    omega : float

        Modal frequency associated with the separated
        homogeneous mode.

    Examples
    --------

    Vertical interface

    >>> InterfaceGeometry(
    ...     interface_type="vertical",
    ...     location=64.0,
    ...     decay_rate=alpha,
    ...     omega=omega_y,
    ... )

    Horizontal interface

    >>> InterfaceGeometry(
    ...     interface_type="horizontal",
    ...     location=20.0,
    ...     decay_rate=beta,
    ...     omega=omega_x,
    ... )
    """

    interface_type: str
    location: float
    decay_rate: float | None = None
    omega: float | None = None
    
    
    # multi-mode API
    decay_rates: np.ndarray | None = None
    omegas: np.ndarray | None = None
    
    def get_modes(self):

        if self.decay_rates is not None:

            return (
                np.asarray(self.decay_rates),
                np.asarray(self.omegas),
            )

        return (
            np.asarray([self.decay_rate]),
            np.asarray([self.omega]),
        )


# ==========================================================
# CORRECTION RESULT
# ==========================================================

@dataclass
class CorrectionResult:
    """
    Output returned by InterfaceCorrector.correct().

    Attributes
    ----------
    coefficient_A : float

        Least-squares amplitude associated with domain A.

    coefficient_B : float

        Least-squares amplitude associated with domain B.

    correction_A : ndarray

        Reconstructed homogeneous field on domain A.

    correction_B : ndarray

        Reconstructed homogeneous field on domain B.

    corrected_A : ndarray

        Corrected finite-element solution

            uA - correction_A

    corrected_B : ndarray

        Corrected finite-element solution

            uB - correction_B

    error_before_A : float

        L2 error of domain A before reconstruction.

    error_before_B : float

        L2 error of domain B before reconstruction.

    error_after_A : float

        L2 error of domain A after reconstruction.

    error_after_B : float

        L2 error of domain B after reconstruction.

    Examples
    --------

    >>> result = InterfaceCorrector.correct(...)
    >>> print(result.coefficient_A)
    >>> print(result.error_after_A)
    """

    coefficient_A: np.ndarray
    coefficient_B: np.ndarray

    correction_A: np.ndarray
    correction_B: np.ndarray

    corrected_A: np.ndarray
    corrected_B: np.ndarray

    error_before_A: float
    error_before_B: float

    error_after_A: float
    error_after_B: float


# ==========================================================
# INTERFACE CORRECTOR
# ==========================================================

class InterfaceCorrector:
    """
    Generic implementation of the interface reconstruction method.

    This class encapsulates the complete workflow

        1. Select fitting planes
        2. Compute residuals
        3. Build least-squares system
        4. Compute modal amplitudes
        5. Reconstruct homogeneous fields
        6. Correct finite-element solutions
        7. Evaluate reconstruction quality

    The implementation may be reused recursively for hierarchical
    domain decomposition.

    Examples
    --------

    >>> result = InterfaceCorrector.correct(
    ...     basisA,
    ...     uA,
    ...     basisB,
    ...     uB,
    ...     interp_reference,
    ...     geometry,
    ...     logger,
    ... )
    """

    # ======================================================
    # VERTICAL FIT
    # ======================================================

    @staticmethod
    
    def _fit_vertical(
        basisA,
        uA,
        basisB,
        uB,
        interp_ref,
        xI,
        alphas,
        omegas,
        logger=None,
    ):

        """
        Compute least-squares amplitudes for a vertical interface.

        Interface geometry

            Domain A | Domain B

        with

            x = xI

        Matching planes are selected automatically:

            x0 = third-to-last plane in domain A
            x1 = third plane in domain B

        Parameters
        ----------
        basisA, basisB
            Finite-element bases.

        uA, uB
            Finite-element solution vectors.

        interp_ref
            Reference solution interpolator.

        xI
            Interface position.

        alpha
            Exponential decay rate.

        omega
            Modal frequency.

        logger
            Logger instance.

        Returns
        -------
        (AA, AB)

            Fitted amplitudes for domains A and B.
        """

        
        if logger is not None:

            logger.info(
                "Starting vertical least-squares fit"
            )

            logger.debug(
                f"Interface location xI = {xI:.6f}"
            )

            logger.debug(
                f"Number of modes = {len(omegas)}"
            )

        XA, YA = basisA.doflocs
        XB, YB = basisB.doflocs

        x0 = np.unique(XA)[-3]
        x1 = np.unique(XB)[2]

        if logger is not None:

            logger.debug(
                f"Matching plane A : x0={x0:.6f}"
            )

            logger.debug(
                f"Matching plane B : x1={x1:.6f}"
            )

        idxA = np.where(
            np.abs(XA - x0) < 1.0e-12
        )[0]

        idxB = np.where(
            np.abs(XB - x1) < 1.0e-12
        )[0]

        if logger is not None:

            logger.debug(
                f"Points on A fitting plane = {len(idxA)}"
            )

            logger.debug(
                f"Points on B fitting plane = {len(idxB)}"
            )

        nmodes = len(omegas)

        rows = []
        rhs = []

        # ==========================================
        # DOMAIN A
        # ==========================================

        for idx in idxA:

            y = YA[idx]

            row = []

            for alpha, omega in zip(
                alphas,
                omegas,
            ):

                row.append(
                    np.exp(
                        -alpha * (xI - x0)
                    )
                    *
                    np.sin(
                        omega * y
                    )
                )

            row.extend([0.0] * nmodes)

            rows.append(row)

            rhs.append(
                uA[idx]
                -
                interp_ref(
                    np.array([x0]),
                    np.array([y]),
                )[0]
            )

        # ==========================================
        # DOMAIN B
        # ==========================================

        for idx in idxB:

            y = YB[idx]

            row = [0.0] * nmodes

            for alpha, omega in zip(
                alphas,
                omegas,
            ):

                row.append(
                    np.exp(
                        -alpha * (x1 - xI)
                    )
                    *
                    np.sin(
                        omega * y
                    )
                )

            rows.append(row)

            rhs.append(
                uB[idx]
                -
                interp_ref(
                    np.array([x1]),
                    np.array([y]),
                )[0]
            )

        M = np.asarray(rows)

        rhs = np.asarray(rhs)

        coef, *_ = np.linalg.lstsq(
            M,
            rhs,
            rcond=None,
        )

        AA = coef[:nmodes]

        AB = coef[nmodes:]

        if logger is not None:

            logger.info(
                "Vertical least-squares fit completed"
            )

            logger.info(
                f"AA = {AA}"
            )

            logger.info(
                f"AB = {AB}"
            )

        return AA, AB


    # ======================================================
    # HORIZONTAL FIT
    # ======================================================

    @staticmethod
    def _fit_horizontal(
        basisA,
        uA,
        basisB,
        uB,
        interp_ref,
        yI,
        betas,
        omegas,
        logger=None,
    ):
        """
        Compute least-squares amplitudes for a horizontal interface.

        Interface geometry

                Domain B
            ----------------
                Domain A

        with

                y = yI

        Returns
        -------
        (AA, AB)

            Fitted amplitudes for domains A and B.
        """

        
        if logger is not None:

            logger.info(
                "Starting horizontal least-squares fit"
            )

            logger.debug(
                f"Interface location yI = {yI:.6f}"
            )

            logger.debug(
                f"Number of modes = {len(omegas)}"
            )

        XA, YA = basisA.doflocs
        XB, YB = basisB.doflocs

        y0 = np.unique(YA)[-3]
        y1 = np.unique(YB)[2]

        idxA = np.where(
            np.abs(YA - y0) < 1.0e-12
        )[0]

        idxB = np.where(
            np.abs(YB - y1) < 1.0e-12
        )[0]

        nmodes = len(omegas)

        rows = []
        rhs = []

        # ==========================================
        # DOMAIN A
        # ==========================================

        for idx in idxA:

            x = XA[idx]

            row = []

            for beta, omega in zip(
                betas,
                omegas,
            ):

                row.append(
                    np.exp(
                        -beta * (yI - y0)
                    )
                    *
                    np.sin(
                        omega * x
                    )
                )

            row.extend([0.0] * nmodes)

            rows.append(row)

            rhs.append(
                uA[idx]
                -
                interp_ref(
                    np.array([x]),
                    np.array([y0]),
                )[0]
            )

        # ==========================================
        # DOMAIN B
        # ==========================================

        for idx in idxB:

            x = XB[idx]

            row = [0.0] * nmodes

            for beta, omega in zip(
                betas,
                omegas,
            ):

                row.append(
                    np.exp(
                        -beta * (y1 - yI)
                    )
                    *
                    np.sin(
                        omega * x
                    )
                )

            rows.append(row)

            rhs.append(
                uB[idx]
                -
                interp_ref(
                    np.array([x]),
                    np.array([y1]),
                )[0]
            )
            
        

        M = np.asarray(rows)

        rhs = np.asarray(rhs)

        coef, *_ = np.linalg.lstsq(
            M,
            rhs,
            rcond=None,
        )

        AA = coef[:nmodes]

        AB = coef[nmodes:]

        if logger is not None:

            logger.info(
                "Horizontal least-squares fit completed"
            )

            logger.info(
                f"AA = {AA}"
            )

            logger.info(
                f"AB = {AB}"
            )

        return AA, AB


    # ======================================================
    # MAIN DRIVER
    # ======================================================

    @staticmethod
    def correct(
        basisA,
        uA,
        basisB,
        uB,
        interp_reference,
        geometry,
        logger=None,
    ):
        """
        Apply interface correction between two neighbouring domains.

        This is the main public entry point of the module.

        Parameters
        ----------
        basisA, basisB
            Finite-element basis objects.

        uA, uB
            Finite-element solution vectors.

        interp_reference
            Reference solution interpolator.

        geometry : InterfaceGeometry
            Interface description.

        logger
            Optional logger.

        Returns
        -------
        CorrectionResult
        """

        if logger is not None:
            logger.info(
                f"Applying {geometry.interface_type} "
                "interface correction"
            )

        if geometry.interface_type == "vertical":

            alphas, omegas = geometry.get_modes()

            cA, cB = InterfaceCorrector._fit_vertical(
                basisA,
                uA,
                basisB,
                uB,
                interp_reference,
                geometry.location,
                alphas,
                omegas,
                logger,
            )

            XA, YA = basisA.doflocs
            XB, YB = basisB.doflocs

            eA = np.zeros_like(uA)
            eB = np.zeros_like(uB)

            for A, alpha, omega in zip(
                cA,
                alphas,
                omegas,
            ):

                eA += (

                    A

                    * np.exp(
                        -alpha
                        * (
                            geometry.location
                            - XA
                        )
                    )

                    * np.sin(
                        omega
                        * YA
                    )
                )

            for B, alpha, omega in zip(
                cB,
                alphas,
                omegas,
            ):

                eB += (

                    B

                    * np.exp(
                        -alpha
                        * (
                            XB
                            - geometry.location
                        )
                    )

                    * np.sin(
                        omega
                        * YB
                    )
                )

        else:

            betas, omegas = geometry.get_modes()

            cA, cB = InterfaceCorrector._fit_horizontal(
                basisA,
                uA,
                basisB,
                uB,
                interp_reference,
                geometry.location,
                betas,
                omegas,
                logger,
            )

            XA, YA = basisA.doflocs
            XB, YB = basisB.doflocs

            eA = np.zeros_like(uA)
            eB = np.zeros_like(uB)

            for A, beta, omega in zip(
                cA,
                betas,
                omegas,
            ):

                eA += (

                    A

                    * np.exp(
                        -beta
                        * (
                            geometry.location
                            - YA
                        )
                    )

                    * np.sin(
                        omega
                        * XA
                    )
                )

            for B, beta, omega in zip(
                cB,
                betas,
                omegas,
            ):

                eB += (

                    B

                    * np.exp(
                        -beta
                        * (
                            YB
                            - geometry.location
                        )
                    )

                    * np.sin(
                        omega
                        * XB
                    )
                )

        if logger is not None:
            logger.info("Constructing corrected solutions")

        uAc = uA - eA
        uBc = uB - eB

        refA = interp_reference(XA, YA)
        refB = interp_reference(XB, YB)

        error_before_A = np.linalg.norm(uA - refA)
        error_before_B = np.linalg.norm(uB - refB)

        error_after_A = np.linalg.norm(uAc - refA)
        error_after_B = np.linalg.norm(uBc - refB)

        if logger is not None:

            logger.info(
                f"Domain A : "
                f"{error_before_A:.6e} -> "
                f"{error_after_A:.6e}"
            )

            logger.info(
                f"Domain B : "
                f"{error_before_B:.6e} -> "
                f"{error_after_B:.6e}"
            )

            if error_after_A > 0:
                logger.info(
                    f"Domain A improvement = "
                    f"{error_before_A/error_after_A:.3f}"
                )

            if error_after_B > 0:
                logger.info(
                    f"Domain B improvement = "
                    f"{error_before_B/error_after_B:.3f}"
                )

        return CorrectionResult(
            coefficient_A=cA,
            coefficient_B=cB,
            correction_A=eA,
            correction_B=eB,
            corrected_A=uAc,
            corrected_B=uBc,
            error_before_A=error_before_A,
            error_before_B=error_before_B,
            error_after_A=error_after_A,
            error_after_B=error_after_B,
        )


if __name__ == "__main__":
    print("Generic interface correction framework.")