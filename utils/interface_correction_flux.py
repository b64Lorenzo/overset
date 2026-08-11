#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass
class FluxSubdomain:

    id: int

    basis: Any

    solution: np.ndarray


@dataclass
class FluxInterface:

    id: int

    minus_subdomain: int

    plus_subdomain: int

    interface_type: str

    location: float

    Lx: float

    Ly: float

    reaction_coefficient: float

    nmodes: int = 8


@dataclass
class FluxCorrectionResult:

    amplitudes: Dict[int, np.ndarray]

    corrections: Dict[int, np.ndarray]

    corrected_solutions: Dict[int, np.ndarray]

    residual_before_solution: float

    residual_before_flux: float

    residual_after_solution: float

    residual_after_flux: float


# ==========================================================
# CORRECTOR
# ==========================================================

class FluxInterfaceCorrector:

    @staticmethod
    def correct(
        subdomains,
        interfaces,
        F,
        logger=None,
    ):

        corrections = {}

        corrected = {}

        for sd in subdomains:

            corrections[sd.id] = np.zeros_like(
                sd.solution
            )

            corrected[sd.id] = sd.solution.copy()

        amplitudes = {}

        before_sol = 0.0
        before_flux = 0.0

        after_sol = 0.0
        after_flux = 0.0

        # =============================================
        # PROCESS EACH INTERFACE
        # =============================================

        for interface in interfaces:

            sd_minus = (
                FluxInterfaceCorrector
                ._find_subdomain(
                    subdomains,
                    interface.minus_subdomain
                )
            )

            sd_plus = (
                FluxInterfaceCorrector
                ._find_subdomain(
                    subdomains,
                    interface.plus_subdomain
                )
            )

            Ju, Jq, s = (
                FluxInterfaceCorrector
                ._compute_interface_jumps(
                    interface,
                    corrected[sd_minus.id],
                    corrected[sd_plus.id],
                    sd_minus.basis,
                    sd_plus.basis,
                    F
                )
            )

            before_sol += np.linalg.norm(Ju)

            before_flux += np.linalg.norm(Jq)

            amps_minus, amps_plus = (
                FluxInterfaceCorrector
                ._fit_interface_modes(
                    Ju,
                    Jq,
                    s,
                    interface,
                    F,
                )
            )

            amplitudes[
                interface.id
            ] = (
                amps_minus,
                amps_plus
            )

            corr_minus = (
                FluxInterfaceCorrector
                ._build_correction(
                    sd_minus.basis,
                    interface,
                    amps_minus,
                    side="minus",
                    F = F
                )
            )

            corr_plus = (
                FluxInterfaceCorrector
                ._build_correction(
                    sd_plus.basis,
                    interface,
                    amps_plus,
                    side="plus",
                    F = F
                )
            )

            corrections[
                sd_minus.id
            ] += corr_minus

            corrections[
                sd_plus.id
            ] += corr_plus

            corrected[
                sd_minus.id
            ] -= corr_minus

            corrected[
                sd_plus.id
            ] -= corr_plus

            Ju2, Jq2, _ = (
                FluxInterfaceCorrector
                ._compute_interface_jumps(
                    interface,
                    corrected[sd_minus.id],
                    corrected[sd_plus.id],
                    sd_minus.basis,
                    sd_plus.basis,
                    F
                )
            )

            after_sol += np.linalg.norm(
                Ju2
            )

            after_flux += np.linalg.norm(
                Jq2
            )

        if logger:

            logger.info(
                f"solution before = "
                f"{before_sol:.6e}"
            )

            logger.info(
                f"solution after = "
                f"{after_sol:.6e}"
            )

            logger.info(
                f"flux before = "
                f"{before_flux:.6e}"
            )

            logger.info(
                f"flux after = "
                f"{after_flux:.6e}"
            )

        return FluxCorrectionResult(

            amplitudes=amplitudes,

            corrections=corrections,

            corrected_solutions=corrected,

            residual_before_solution=
            before_sol,

            residual_before_flux=
            before_flux,

            residual_after_solution=
            after_sol,

            residual_after_flux=
            after_flux,
        )

    # ======================================================
    # BUILD ANALYTICAL MODES
    # ======================================================

    @staticmethod
    def _build_correction(
        basis,
        interface,
        amps,
        side,
        F,
    ):

        X, Y = basis.doflocs

        corr = np.zeros(
            len(X)
        )

        if interface.interface_type == "vertical":

            Lt = interface.Ly

            d = (
                interface.location - X
                if side == "minus"
                else X - interface.location
            )

            tangential = Y

        else:

            Lt = interface.Lx

            d = (
                interface.location - Y
                if side == "minus"
                else Y - interface.location
            )

            tangential = X
        
        d = np.maximum(d, 0.0)
        
        for n in range(
            1,
            len(amps) + 1
        ):

            alpha = np.sqrt(
                interface.reaction_coefficient / F
                +
                (
                    n * np.pi / Lt
                )**2
            )

            phi = (
                np.sin(
                    n*np.pi*tangential/Lt
                )
                *
                np.exp(
                    -alpha*d
                )
            )

            corr += amps[n-1] * phi

        return corr

    # ======================================================

    @staticmethod
    def _find_subdomain(
        subdomains,
        sid
    ):

        for sd in subdomains:

            if sd.id == sid:
                return sd

        raise RuntimeError(
            f"Subdomain {sid} not found"
        )

    # ======================================================

    @staticmethod
    def _compute_interface_jumps(
        interface,
        u_minus,
        u_plus,
        basis_minus,
        basis_plus,
        F,
    ):

        idx_minus = (
            FluxInterfaceCorrector
            ._interface_indices(
                basis_minus,
                interface
            )
        )

        idx_plus = (
            FluxInterfaceCorrector
            ._interface_indices(
                basis_plus,
                interface
            )
        )

        Xm, Ym = basis_minus.doflocs

        if interface.interface_type == "vertical":

            order = np.argsort(
                Ym[idx_minus]
            )

            s = Ym[idx_minus][order]

        else:

            order = np.argsort(
                Xm[idx_minus]
            )

            s = Xm[idx_minus][order]

        idx_minus = idx_minus[order]
        idx_plus = idx_plus[order]

        Ju = (
            u_minus[idx_minus]
            -
            u_plus[idx_plus]
        )

        q_minus = (
            FluxInterfaceCorrector
            ._compute_flux(
                basis_minus,
                u_minus,
                idx_minus,
                F,
                interface.interface_type,
                "minus"
            )
        )

        q_plus = (
            FluxInterfaceCorrector
            ._compute_flux(
                basis_plus,
                u_plus,
                idx_plus,
                F,
                interface.interface_type,
                "plus"
            )
        )

        Jq = q_minus - q_plus

        return Ju, Jq, s

    # ======================================================

    @staticmethod
    def _interface_indices(
        basis,
        interface
    ):

        X, Y = basis.doflocs

        tol = 1e-8

        if interface.interface_type == "vertical":

            return np.where(
                np.abs(
                    X-interface.location
                ) < tol
            )[0]

        if interface.interface_type == "horizontal":

            return np.where(
                np.abs(
                    Y-interface.location
                ) < tol
            )[0]

        raise RuntimeError(
            "Unknown interface type"
        )

    # ======================================================

    @staticmethod
    def _compute_flux(
        basis,
        u,
        interface_idx,
        F,
        interface_type,
        side,
    ):

        X, Y = basis.doflocs

        flux = []

        for idx in interface_idx:

            if interface_type == "vertical":

                y = Y[idx]

                row = np.where(
                    np.abs(Y-y)
                    < 1e-10
                )[0]

                row = row[
                    np.argsort(
                        X[row]
                    )
                ]

                pos = np.where(
                    row == idx
                )[0][0]

                try:

                    if side == "minus":

                        i0=row[pos-2]
                        i1=row[pos-1]
                        i2=row[pos]

                        dx=X[i2]-X[i1]

                        grad=(
                            3*u[i2]
                            -4*u[i1]
                            +u[i0]
                        )/(2*dx)

                    else:

                        i0=row[pos]
                        i1=row[pos+1]
                        i2=row[pos+2]

                        dx=X[i1]-X[i0]

                        grad=(
                            -3*u[i0]
                            +4*u[i1]
                            -u[i2]
                        )/(2*dx)

                except:

                    grad=0.0

            else:

                x = X[idx]

                col = np.where(
                    np.abs(X-x)
                    < 1e-10
                )[0]

                col = col[
                    np.argsort(
                        Y[col]
                    )
                ]

                pos = np.where(
                    col == idx
                )[0][0]

                try:

                    if side == "minus":

                        i0=col[pos-2]
                        i1=col[pos-1]
                        i2=col[pos]

                        dy=Y[i2]-Y[i1]

                        grad=(
                            3*u[i2]
                            -4*u[i1]
                            +u[i0]
                        )/(2*dy)

                    else:

                        i0=col[pos]
                        i1=col[pos+1]
                        i2=col[pos+2]

                        dy=Y[i1]-Y[i0]

                        grad=(
                            -3*u[i0]
                            +4*u[i1]
                            -u[i2]
                        )/(2*dy)

                except:

                    grad=0.0

            flux.append(
                F*grad
            )

        return np.asarray(flux)
    
    @staticmethod
    def _fit_interface_modes(
        Ju,
        Jq,
        s,
        interface,
        F,
    ):

        Lt = (
            interface.Ly
            if interface.interface_type == "vertical"
            else interface.Lx
        )

        nmodes = interface.nmodes

        rows = []
        rhs = []

        # ==========================================
        # SOLUTION JUMP
        # ==========================================

        for i, si in enumerate(s):

            row = []

            # minus amplitudes

            for n in range(1, nmodes + 1):

                psi = np.sin(
                    n*np.pi*si/Lt
                )

                row.append(+psi)

            # plus amplitudes

            for n in range(1, nmodes + 1):

                psi = np.sin(
                    n*np.pi*si/Lt
                )

                row.append(-psi)

            rows.append(row)

            rhs.append(Ju[i])

        # ==========================================
        # FLUX JUMP
        # ==========================================

        for i, si in enumerate(s):

            row = []

            # minus amplitudes

            for n in range(1, nmodes + 1):

                alpha = np.sqrt(
                    interface.reaction_coefficient/F
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                psi = np.sin(
                    n*np.pi*si/Lt
                )

                row.append(
                    F * alpha * psi
                )

            # plus amplitudes

            for n in range(1, nmodes + 1):

                alpha = np.sqrt(
                    interface.reaction_coefficient/F
                    +
                    (
                        n*np.pi/Lt
                    )**2
                )

                psi = np.sin(
                    n*np.pi*si/Lt
                )

                row.append(
                    F * alpha * psi
                )

            rows.append(row)

            rhs.append(Jq[i])

        # ==========================================
        # BUILD SYSTEM
        # ==========================================

        print(
            "unique row lengths =",
            np.unique([len(r) for r in rows])
        )

        M = np.asarray(rows)

        rhs = np.asarray(rhs)

        coef, *_ = np.linalg.lstsq(
            M,
            rhs,
            rcond=None,
        )

        amps_minus = coef[:nmodes]

        amps_plus = coef[nmodes:]

        return (
            amps_minus,
            amps_plus,
        )