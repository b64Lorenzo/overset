#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any

import numpy as np


# ==========================================================
# GEOMETRY
# ==========================================================

@dataclass
class ModalSolutionGeometry:

    interface_type: str

    location: float

    Lx: float

    Ly: float

    reaction_coefficient: float

    diffusion_coefficient: float

    nmodes: int = 10

    observation_offset: float = 1.0


# ==========================================================
# RESULT
# ==========================================================

@dataclass
class ModalSolutionResult:

    coeffA: np.ndarray

    coeffB: np.ndarray

    correction_A: np.ndarray

    correction_B: np.ndarray

    corrected_A: np.ndarray

    corrected_B: np.ndarray

    error_before_A: float

    error_after_A: float

    error_before_B: float

    error_after_B: float


# ==========================================================
# MODAL SOLUTION CORRECTOR
# ==========================================================

class ModalSolutionCorrector:

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

        XA, YA = basisA.doflocs
        XB, YB = basisB.doflocs

        nmodes = geometry.nmodes

        coeffA = np.zeros(nmodes)
        coeffB = np.zeros(nmodes)

        if geometry.interface_type == "vertical":

            xI = geometry.location

            x0 = xI - geometry.observation_offset
            x1 = xI + geometry.observation_offset

            xA_actual = XA[
                np.argmin(
                    np.abs(XA - x0)
                )
            ]

            xB_actual = XB[
                np.argmin(
                    np.abs(XB - x1)
                )
            ]

            idxA = np.where(
                np.abs(XA - xA_actual)
                < 1e-10
            )[0]

            idxB = np.where(
                np.abs(XB - xB_actual)
                < 1e-10
            )[0]

            if len(idxA) == 0:
                raise RuntimeError(
                    f"No observation slice found near x={x0}"
                )

            if len(idxB) == 0:
                raise RuntimeError(
                    f"No observation slice found near x={x1}"
                )

            idxA = idxA[np.argsort(YA[idxA])]
            idxB = idxB[np.argsort(YB[idxB])]

            sA = YA[idxA]
            sB = YB[idxB]

            Lt = geometry.Ly

            refA = interp_reference(
                XA[idxA],
                YA[idxA]
            )

            refB = interp_reference(
                XB[idxB],
                YB[idxB]
            )

            dA = (
                uA[idxA]
                -
                refA
            )

            dB = (
                uB[idxB]
                -
                refB
            )

            A_matrix = np.zeros(
                (
                    len(idxA),
                    nmodes
                )
            )

            B_matrix = np.zeros(
                (
                    len(idxB),
                    nmodes
                )
            )

            for n in range(
                1,
                nmodes + 1
            ):

                alpha = np.sqrt(
                    geometry.reaction_coefficient
                    /
                    geometry.diffusion_coefficient
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                A_matrix[:, n-1] = (
                    np.sin(
                        n*np.pi*sA/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            xI - x0
                        )
                    )
                )

                B_matrix[:, n-1] = (
                    np.sin(
                        n*np.pi*sB/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            x1 - xI
                        )
                    )
                )

            coeffA, *_ = np.linalg.lstsq(
                A_matrix,
                dA,
                rcond=None
            )

            coeffB, *_ = np.linalg.lstsq(
                B_matrix,
                dB,
                rcond=None
            )

            corrA = np.zeros_like(uA)
            corrB = np.zeros_like(uB)

            for n in range(
                1,
                nmodes + 1
            ):

                alpha = np.sqrt(
                    geometry.reaction_coefficient
                    /
                    geometry.diffusion_coefficient
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                phiA = (
                    np.sin(
                        n*np.pi*YA/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            xI - XA
                        )
                    )
                )

                phiB = (
                    np.sin(
                        n*np.pi*YB/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            XB - xI
                        )
                    )
                )

                corrA += (
                    coeffA[n-1]
                    *
                    phiA
                )

                corrB += (
                    coeffB[n-1]
                    *
                    phiB
                )

        elif geometry.interface_type == "horizontal":

            yI = geometry.location

            y0 = yI - geometry.observation_offset
            y1 = yI + geometry.observation_offset

            xA_actual = XA[
                np.argmin(
                    np.abs(XA - x0)
                )
            ]

            xB_actual = XB[
                np.argmin(
                    np.abs(XB - x1)
                )
            ]

            idxA = np.where(
                np.abs(XA - xA_actual)
                < 1e-10
            )[0]

            idxB = np.where(
                np.abs(XB - xB_actual)
                < 1e-10
            )[0]

            if len(idxA) == 0:
                raise RuntimeError(
                    f"No observation slice found near x={x0}"
                )

            if len(idxB) == 0:
                raise RuntimeError(
                    f"No observation slice found near x={x1}"
                )

            idxA = idxA[np.argsort(XA[idxA])]
            idxB = idxB[np.argsort(XB[idxB])]

            sA = XA[idxA]
            sB = XB[idxB]

            Lt = geometry.Lx

            refA = interp_reference(
                XA[idxA],
                YA[idxA]
            )

            refB = interp_reference(
                XB[idxB],
                YB[idxB]
            )

            dA = (
                uA[idxA]
                -
                refA
            )

            dB = (
                uB[idxB]
                -
                refB
            )

            A_matrix = np.zeros(
                (
                    len(idxA),
                    nmodes
                )
            )

            B_matrix = np.zeros(
                (
                    len(idxB),
                    nmodes
                )
            )

            for n in range(
                1,
                nmodes + 1
            ):

                alpha = np.sqrt(
                    geometry.reaction_coefficient
                    /
                    geometry.diffusion_coefficient
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                A_matrix[:, n-1] = (
                    np.sin(
                        n*np.pi*sA/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            yI - y0
                        )
                    )
                )

                B_matrix[:, n-1] = (
                    np.sin(
                        n*np.pi*sB/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            y1 - yI
                        )
                    )
                )

            coeffA, *_ = np.linalg.lstsq(
                A_matrix,
                dA,
                rcond=None
            )

            coeffB, *_ = np.linalg.lstsq(
                B_matrix,
                dB,
                rcond=None
            )

            corrA = np.zeros_like(uA)
            corrB = np.zeros_like(uB)

            for n in range(
                1,
                nmodes + 1
            ):

                alpha = np.sqrt(
                    geometry.reaction_coefficient
                    /
                    geometry.diffusion_coefficient
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                phiA = (
                    np.sin(
                        n*np.pi*XA/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            yI - YA
                        )
                    )
                )

                phiB = (
                    np.sin(
                        n*np.pi*XB/Lt
                    )
                    *
                    np.exp(
                        -alpha
                        *
                        (
                            YB - yI
                        )
                    )
                )

                corrA += (
                    coeffA[n-1]
                    *
                    phiA
                )

                corrB += (
                    coeffB[n-1]
                    *
                    phiB
                )

        else:

            raise RuntimeError(
                "Unknown interface type"
            )

        uAc = uA - corrA
        uBc = uB - corrB

        refA_full = interp_reference(
            XA,
            YA
        )

        refB_full = interp_reference(
            XB,
            YB
        )

        err_before_A = np.linalg.norm(
            uA - refA_full
        )

        err_after_A = np.linalg.norm(
            uAc - refA_full
        )

        err_before_B = np.linalg.norm(
            uB - refB_full
        )

        err_after_B = np.linalg.norm(
            uBc - refB_full
        )

        if logger is not None:

            logger.info(
                f"nmodes = {nmodes}"
            )

            logger.info(
                f"Left before  = "
                f"{err_before_A:.6e}"
            )

            logger.info(
                f"Left after   = "
                f"{err_after_A:.6e}"
            )

            logger.info(
                f"Right before = "
                f"{err_before_B:.6e}"
            )

            logger.info(
                f"Right after  = "
                f"{err_after_B:.6e}"
            )

        return ModalSolutionResult(

            coeffA=coeffA,

            coeffB=coeffB,

            correction_A=corrA,

            correction_B=corrB,

            corrected_A=uAc,

            corrected_B=uBc,

            error_before_A=err_before_A,

            error_after_A=err_after_A,

            error_before_B=err_before_B,

            error_after_B=err_after_B,
        )