"""
overlap1d.py
============

Author
------
Lorenzo Zambelli

Version
-------
1.0

Description
-----------
Implementation of the one-dimensional overlap-based reconstruction
methodology described in the "Overlapping Domains" section of the report.

The reconstruction exploits the superposition principle

    u = u_p + u_h

and assumes that the mismatch between independently computed subdomain
solutions is primarily associated with homogeneous corrections generated
by the artificial interface conditions.

The correction is represented as

    u_h^W = a_W φ_W
    u_h^E = a_E φ_E

where

    φ_W : western homogeneous basis function
    φ_E : eastern homogeneous basis function
    a_W : western homogeneous amplitude
    a_E : eastern homogeneous amplitude

The amplitudes are obtained by solving

    H a = d

with

    d = u_W - u_E.

The module also provides diagnostics such as residual evaluation,
condition-number estimation, and blending utilities.

References
----------
Olver (2014)

Smith et al. (1996)

Chesshire and Henshaw (1990)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class ReconstructionResult:
    """
    Container storing reconstruction diagnostics.

    Attributes
    ----------
    amplitudes : ndarray
        Reconstructed amplitude vector

            [a_W, a_E]^T.

    matrix : ndarray
        Reconstruction matrix H.

    rhs : ndarray
        Observation vector d.

    residual : ndarray
        Residual

            d - H a.

    condition_number : float
        Condition number of H.

    rank : int
        Matrix rank.

    correction_west : ndarray
        Reconstructed western homogeneous correction.

    correction_east : ndarray
        Reconstructed eastern homogeneous correction.
    """

    amplitudes: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    residual: np.ndarray
    condition_number: float
    rank: int
    correction_west: np.ndarray
    correction_east: np.ndarray


# ============================================================
# RECONSTRUCTOR
# ============================================================

class OverlapReconstructor1D:
    """
    Overlap-based homogeneous reconstruction.

    The reconstruction seeks amplitudes a_W and a_E
    such that

        u_W - u_E
        =
        a_W φ_W
        -
        a_E φ_E.

    Notes
    -----
    The implementation follows the notation adopted in
    Chapter 2 of the report.
    """

    # --------------------------------------------------------
    # BASIS FUNCTIONS
    # --------------------------------------------------------

    @staticmethod
    def phi_west(
        x,
        x0,
        xE,
        lam,
    ):
        """
        Western normalized basis function.

        Parameters
        ----------
        x : ndarray
            Evaluation coordinates.

        x0 : float
            Physical western boundary.

        xE : float
            Eastern edge of overlap region.

        lam : float
            Decay parameter λ.

        Returns
        -------
        ndarray
            Basis function satisfying

                φ_W(x0)=0
                φ_W(xE)=1
        """

        return (
            np.sinh(
                lam * (x - x0)
            )
            /
            np.sinh(
                lam * (xE - x0)
            )
        )

    @staticmethod
    def phi_east(
        x,
        xN,
        xW,
        lam,
    ):
        """
        Eastern normalized basis function.

        Parameters
        ----------
        x : ndarray
            Evaluation coordinates.

        xN : float
            Physical eastern boundary.

        xW : float
            Western edge of overlap region.

        lam : float
            Decay parameter λ.

        Returns
        -------
        ndarray
            Basis function satisfying

                φ_E(xN)=0
                φ_E(xW)=1
        """

        return (
            np.sinh(
                lam * (xN - x)
            )
            /
            np.sinh(
                lam * (xN - xW)
            )
        )

    # --------------------------------------------------------
    # SYSTEM ASSEMBLY
    # --------------------------------------------------------

    def assemble_system(
        self,
        x_obs,
        uW_obs,
        uE_obs,
        x0,
        xN,
        xW,
        xE,
        lam,
    ):
        """
        Assemble the reconstruction system

            H a = d.

        Parameters
        ----------
        x_obs : ndarray
            Observation coordinates.

        uW_obs : ndarray
            West-domain observations.

        uE_obs : ndarray
            East-domain observations.

        x0, xN : float
            Physical domain boundaries.

        xW, xE : float
            Overlap boundaries.

        lam : float
            Decay parameter.

        Returns
        -------
        H : ndarray
            Reconstruction matrix.

        d : ndarray
            Observation vector.
        """

        phiW = self.phi_west(
            x_obs,
            x0,
            xE,
            lam,
        )

        phiE = self.phi_east(
            x_obs,
            xN,
            xW,
            lam,
        )

        H = np.column_stack(
            [
                phiW,
                -phiE,
            ]
        )

        d = (
            np.asarray(uW_obs)
            -
            np.asarray(uE_obs)
        )

        return (
            H,
            d,
            phiW,
            phiE,
        )

    # --------------------------------------------------------
    # SOLVER
    # --------------------------------------------------------

    def solve(
        self,
        x_obs,
        uW_obs,
        uE_obs,
        x0,
        xN,
        xW,
        xE,
        lam,
    ):
        """
        Solve the overlap reconstruction problem.

        Parameters
        ----------
        x_obs : ndarray
            Observation coordinates.

        uW_obs : ndarray
            West-domain observations.

        uE_obs : ndarray
            East-domain observations.

        x0, xN : float
            Physical boundaries.

        xW, xE : float
            Overlap boundaries.

        lam : float
            Decay parameter.

        Returns
        -------
        ReconstructionResult
            Reconstruction information.
        """

        H, d, phiW, phiE = (
            self.assemble_system(
                x_obs,
                uW_obs,
                uE_obs,
                x0,
                xN,
                xW,
                xE,
                lam,
            )
        )

        amplitudes, *_ = np.linalg.lstsq(
            H,
            d,
            rcond=None,
        )

        residual = (
            d
            -
            H @ amplitudes
        )

        try:
            cond = np.linalg.cond(H)
        except Exception:
            cond = np.inf

        rank = np.linalg.matrix_rank(H)

        aW = amplitudes[0]
        aE = amplitudes[1]

        correction_west = aW * phiW
        correction_east = aE * phiE

        return ReconstructionResult(
            amplitudes=amplitudes,
            matrix=H,
            rhs=d,
            residual=residual,
            condition_number=cond,
            rank=rank,
            correction_west=correction_west,
            correction_east=correction_east,
        )

    # --------------------------------------------------------
    # GLOBAL RECONSTRUCTION
    # --------------------------------------------------------

    @staticmethod
    def reconstruct_homogeneous(
        x,
        amplitude,
        basis,
    ):
        """
        Reconstruct homogeneous correction.

        Parameters
        ----------
        x : ndarray
            Coordinates.

        amplitude : float
            Homogeneous amplitude.

        basis : ndarray
            Basis-function values.

        Returns
        -------
        ndarray
            Homogeneous correction.
        """

        return amplitude * basis

    @staticmethod
    def corrected_solution(
        u,
        correction,
    ):
        """
        Remove homogeneous correction.

        Parameters
        ----------
        u : ndarray
            Local solution.

        correction : ndarray
            Reconstructed homogeneous component.

        Returns
        -------
        ndarray
            Corrected solution.
        """

        return (
            np.asarray(u)
            -
            np.asarray(correction)
        )

    # --------------------------------------------------------
    # BLENDING
    # --------------------------------------------------------

    @staticmethod
    def linear_weight(
        x,
        xW,
        xE,
    ):
        """
        Partition-of-unity weight.

        Parameters
        ----------
        x : ndarray
            Coordinates.

        xW : float
            Western overlap boundary.

        xE : float
            Eastern overlap boundary.

        Returns
        -------
        ndarray
            Linear blending weight.
        """

        return (
            xE - x
        ) / (
            xE - xW
        )

    @staticmethod
    def blend(
        x,
        uW,
        uE,
        xW,
        xE,
    ):
        """
        Blend corrected solutions.

        Parameters
        ----------
        x : ndarray
            Coordinates.

        uW : ndarray
            Corrected west solution.

        uE : ndarray
            Corrected east solution.

        xW : float
            Western overlap boundary.

        xE : float
            Eastern overlap boundary.

        Returns
        -------
        ndarray
            Blended solution.
        """

        omega = (
            OverlapReconstructor1D
            .linear_weight(
                x,
                xW,
                xE,
            )
        )

        return (
            omega * uW
            +
            (1.0 - omega) * uE
        )

    # --------------------------------------------------------
    # ERROR ESTIMATION
    # --------------------------------------------------------

    @staticmethod
    def residual(
        uW,
        uE,
        aW,
        aE,
        phiW,
        phiE,
    ):
        """
        Compute reconstruction residual.

        Parameters
        ----------
        uW : ndarray
            West-domain solution.

        uE : ndarray
            East-domain solution.

        aW, aE : float
            Reconstructed amplitudes.

        phiW, phiE : ndarray
            Basis-function values.

        Returns
        -------
        ndarray
            Residual

                r =
                uW - uE
                -
                (aW φW - aE φE).
        """

        return (
            np.asarray(uW)
            -
            np.asarray(uE)
            -
            (
                aW * phiW
                -
                aE * phiE
            )
        )

    @staticmethod
    def l2_error(
        numerical,
        reference,
    ):
        """
        Compute L2 error.

        Parameters
        ----------
        numerical : ndarray
            Numerical solution.

        reference : ndarray
            Reference solution.

        Returns
        -------
        float
            Euclidean norm.
        """

        return np.linalg.norm(
            np.asarray(numerical)
            -
            np.asarray(reference)
        )

    @staticmethod
    def condition_number(
        H,
    ):
        """
        Compute matrix condition number.

        Parameters
        ----------
        H : ndarray
            Reconstruction matrix.

        Returns
        -------
        float
            cond(H).
        """

        return np.linalg.cond(H)