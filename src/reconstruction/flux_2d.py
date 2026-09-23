#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq

Side = Literal["l","r","t","b"]
BasisFunction = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

class EigenFunctions2D:
    def __init__(self,a,b,c,d,F0,k):
        self.a=a; self.b=b; self.c=c; self.d=d
        self.F0=F0; self.k=k
        self.Lx=b-a; self.Ly=d-c
        self.mu=np.sqrt(k/F0)

    def left_homo(self,n,x,y):
        beta=n*np.pi/self.Ly
        alpha=np.sqrt(self.mu**2+beta**2)
        return np.sinh(alpha*(self.b-x))/np.sinh(alpha*self.Lx)*np.sin(beta*(y-self.c))

    def right_homo(self,n,x,y):
        beta=n*np.pi/self.Ly
        alpha=np.sqrt(self.mu**2+beta**2)
        return np.sinh(alpha*(x-self.a))/np.sinh(alpha*self.Lx)*np.sin(beta*(y-self.c))

    def bottom_homo(self,n,x,y):
        gamma=n*np.pi/self.Lx
        alpha=np.sqrt(self.mu**2+gamma**2)
        return np.sinh(alpha*(self.d-y))/np.sinh(alpha*self.Ly)*np.sin(gamma*(x-self.a))

    def top_homo(self,n,x,y):
        gamma=n*np.pi/self.Lx
        alpha=np.sqrt(self.mu**2+gamma**2)
        return np.sinh(alpha*(y-self.c))/np.sinh(alpha*self.Ly)*np.sin(gamma*(x-self.a))

@dataclass(slots=True)
class Domain2D:
    a: float; b: float; c: float; d: float
    F0: float; k: float
    interfaces: tuple[Side,...]
    n_modes: int = 10

    @property
    def eigenfunctions(self):
        return EigenFunctions2D(self.a,self.b,self.c,self.d,self.F0,self.k)

    def basis_functions(self):
        ef=self.eigenfunctions
        basis=[]
        for side in self.interfaces:
            for n in range(1,self.n_modes+1):
                if side=='l': basis.append(lambda x,y,n=n,ef=ef: ef.left_homo(n,x,y))
                elif side=='r': basis.append(lambda x,y,n=n,ef=ef: ef.right_homo(n,x,y))
                elif side=='t': basis.append(lambda x,y,n=n,ef=ef: ef.top_homo(n,x,y))
                elif side=='b': basis.append(lambda x,y,n=n,ef=ef: ef.bottom_homo(n,x,y))
        return tuple(basis)

    @property
    def n_unknowns(self):
        return len(self.interfaces)*self.n_modes

@dataclass(slots=True)
class DomainSolution2D:
    domain: Domain2D
    solution: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    interpolator: Callable
    gradx: Callable
    grady: Callable

    @property
    def basis(self):
        return self.domain.basis_functions()

    @property
    def n_unknowns(self):
        return self.domain.n_unknowns

@dataclass(slots=True)
class FluxCorrectionResult2D:
    coefficients: NDArray[np.float64]
    corrections: tuple
    corrected: tuple
    matrix: NDArray[np.float64]
    rhs: NDArray[np.float64]

class FluxInterfaceCorrector2D:

    @staticmethod
    def correct(
        domains,
        y_interface,
        x_interface,
    ):

        D1, D2 = domains

        Nm = D1.domain.n_modes

        Ly = (
            D1.domain.d
            -
            D1.domain.c
        )

        n_unknowns = 2 * Nm

        xeval = np.full_like(
            y_interface,
            x_interface,
            dtype=float,
        )

        # =====================================================
        # PHYSICAL JUMPS
        # =====================================================

        Ju = (
            D1.interpolator(
                xeval,
                y_interface,
            )
            -
            D2.interpolator(
                xeval,
                y_interface,
            )
        )

        Jq = (
            D1.domain.F0
            *
            D1.gradx(
                xeval,
                y_interface,
            )
            -
            D2.domain.F0
            *
            D2.gradx(
                xeval,
                y_interface,
            )
        )

        Ju = np.nan_to_num(
            np.asarray(
                Ju,
                dtype=float,
            )
        )

        Jq = np.nan_to_num(
            np.asarray(
                Jq,
                dtype=float,
            )
        )

        y_interface = np.asarray(
            y_interface,
            dtype=float,
        )

        # =====================================================
        # GLOBAL MATRIX
        # =====================================================

        A = np.zeros(
            (
                2 * Nm,
                2 * Nm,
            ),
            dtype=float,
        )

        b = np.zeros(
            2 * Nm,
            dtype=float,
        )

        phiL = (
            D1.domain
            .eigenfunctions
            .right_homo
        )

        phiR = (
            D2.domain
            .eigenfunctions
            .left_homo
        )

        h = 1e-6

        # =====================================================
        # ASSEMBLY
        # =====================================================

        for n in range(
            1,
            Nm + 1,
        ):

            psi_n = np.sin(
                n
                *
                np.pi
                *
                y_interface
                /
                Ly
            )

            # -------------------------------------------------
            # RHS
            # -------------------------------------------------

            b[n - 1] = -(
                2.0
                /
                Ly
                *
                np.trapezoid(
                    Ju
                    *
                    psi_n,
                    y_interface,
                )
            )

            b[
                Nm + n - 1
            ] = -(
                2.0
                /
                Ly
                *
                np.trapezoid(
                    Jq
                    *
                    psi_n,
                    y_interface,
                )
            )

            # -------------------------------------------------
            # MATRIX
            # -------------------------------------------------

            for m in range(
                1,
                Nm + 1,
            ):

                phiLm = phiL(
                    m,
                    x_interface,
                    y_interface,
                )

                phiRm = phiR(
                    m,
                    x_interface,
                    y_interface,
                )

                A[
                    n - 1,
                    m - 1,
                ] = -(
                    2.0
                    /
                    Ly
                    *
                    np.trapezoid(
                        phiLm
                        *
                        psi_n,
                        y_interface,
                    )
                )

                A[
                    n - 1,
                    Nm + m - 1,
                ] = (
                    2.0
                    /
                    Ly
                    *
                    np.trapezoid(
                        phiRm
                        *
                        psi_n,
                        y_interface,
                    )
                )

                dphiLm = (
                    phiL(
                        m,
                        x_interface + h,
                        y_interface,
                    )
                    -
                    phiL(
                        m,
                        x_interface - h,
                        y_interface,
                    )
                ) / (2.0 * h)

                dphiRm = (
                    phiR(
                        m,
                        x_interface + h,
                        y_interface,
                    )
                    -
                    phiR(
                        m,
                        x_interface - h,
                        y_interface,
                    )
                ) / (2.0 * h)

                A[
                    Nm + n - 1,
                    m - 1,
                ] = -(
                    2.0
                    /
                    Ly
                    *
                    np.trapezoid(
                        D1.domain.F0
                        *
                        dphiLm
                        *
                        psi_n,
                        y_interface,
                    )
                )

                A[
                    Nm + n - 1,
                    Nm + m - 1,
                ] = (
                    2.0
                    /
                    Ly
                    *
                    np.trapezoid(
                        D2.domain.F0
                        *
                        dphiRm
                        *
                        psi_n,
                        y_interface,
                    )
                )

        # =====================================================
        # SOLVE GLOBAL SYSTEM
        # =====================================================

        coeffs, *_ = np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )

        coeffA = coeffs[:Nm]

        coeffB = coeffs[Nm:]

        # =====================================================
        # RECONSTRUCT
        # =====================================================

        XL = D1.x
        YL = D1.y

        XR = D2.x
        YR = D2.y

        uhL = np.zeros_like(
            D1.solution,
            dtype=float,
        )

        uhR = np.zeros_like(
            D2.solution,
            dtype=float,
        )

        for n in range(
            1,
            Nm + 1,
        ):

            uhL += (
                coeffA[n - 1]
                *
                phiL(
                    n,
                    XL,
                    YL,
                )
            )

            uhR += (
                coeffB[n - 1]
                *
                phiR(
                    n,
                    XR,
                    YR,
                )
            )

        return FluxCorrectionResult2D(
            coefficients=coeffs,
            corrections=(
                uhL,
                uhR,
            ),
            corrected=(
                D1.solution - uhL,
                D2.solution - uhR,
            ),
            matrix=A,
            rhs=b,
        )