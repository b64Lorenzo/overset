#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np

@dataclass
class InterfaceGeometry:
    interface_type: str
    location: float
    decay_rates: np.ndarray
    omegas: np.ndarray

@dataclass
class CorrectionResult:
    coefficient_A: np.ndarray
    coefficient_B: np.ndarray
    correction_A: np.ndarray
    correction_B: np.ndarray
    corrected_A: np.ndarray
    corrected_B: np.ndarray
    trace_A: np.ndarray
    trace_B: np.ndarray


class InterfaceCorrector:

    @staticmethod
    def _fit_vertical_paper(basisA,uA,basisB,uB,interp_previous_A,interp_previous_B,geometry):
        XA,YA = basisA.doflocs
        XB,YB = basisB.doflocs

        xI = geometry.location
        alphas = np.asarray(geometry.decay_rates)
        omegas = np.asarray(geometry.omegas)

        x0 = np.unique(XA)[-3]
        x1 = np.unique(XB)[2]

        idxA = np.where(np.abs(XA-x0)<1e-12)[0]
        idxB = np.where(np.abs(XB-x1)<1e-12)[0]

        idxA = idxA[np.argsort(YA[idxA])]
        idxB = idxB[np.argsort(YB[idxB])]

        Ny = len(idxA)
        Nm = len(alphas)

        PhiA = np.zeros((Ny,Nm))
        PhiB = np.zeros((Ny,Nm))

        for i,idx in enumerate(idxA):
            y = YA[idx]
            for j,(a,w) in enumerate(zip(alphas,omegas)):
                PhiA[i,j] = np.exp(-a*(xI-x0))*np.sin(w*y)

        for i,idx in enumerate(idxB):
            y = YB[idx]
            for j,(a,w) in enumerate(zip(alphas,omegas)):
                PhiB[i,j] = np.exp(-a*(x1-xI))*np.sin(w*y)

        I = np.eye(Ny)
        Zyy = np.zeros((Ny,Ny))
        Zym = np.zeros((Ny,Nm))

        M = np.block([
            [I,   Zyy, Zym, PhiB],
            [Zyy, I,   PhiA, Zym],
            [Zyy, Zyy, PhiA, Zym],
            [Zyy, Zyy, Zym, PhiB],
        ])

        rhs = []

        for idx in idxA:
            rhs.append(interp_previous_A(np.array([x0]),np.array([YA[idx]]))[0])

        for idx in idxB:
            rhs.append(interp_previous_B(np.array([x1]),np.array([YB[idx]]))[0])

        for idx in idxA:
            prev = interp_previous_A(np.array([x0]),np.array([YA[idx]]))[0]
            rhs.append(uA[idx]-prev)

        for idx in idxB:
            prev = interp_previous_B(np.array([x1]),np.array([YB[idx]]))[0]
            rhs.append(uB[idx]-prev)

        rhs = np.asarray(rhs)

        lam = 1e-6

        Areg = (
            M.T @ M
            +
            lam*np.eye(
                M.shape[1]
            )
        )

        breg = M.T @ rhs

        x = np.linalg.solve(
            Areg,
            breg,
        )

        traceA = x[:Ny]
        traceB = x[Ny:2*Ny]
        coefA = x[2*Ny:2*Ny+Nm]
        coefB = x[2*Ny+Nm:]

        return coefA,coefB,traceA,traceB

    @staticmethod
    def _fit_horizontal_paper(
        basisA,
        uA,
        basisB,
        uB,
        interp_previous_A,
        interp_previous_B,
        geometry,
    ):

        XA, YA = basisA.doflocs
        XB, YB = basisB.doflocs

        yI = geometry.location

        alphas = np.asarray(
            geometry.decay_rates
        )

        omegas = np.asarray(
            geometry.omegas
        )

        y0 = np.unique(YA)[-3]
        y1 = np.unique(YB)[2]

        idxA = np.where(
            np.abs(YA-y0) < 1e-12
        )[0]

        idxB = np.where(
            np.abs(YB-y1) < 1e-12
        )[0]

        idxA = idxA[
            np.argsort(XA[idxA])
        ]

        idxB = idxB[
            np.argsort(XB[idxB])
        ]

        Nx = len(idxA)

        Nm = len(alphas)

        PhiA = np.zeros(
            (Nx, Nm)
        )

        PhiB = np.zeros(
            (Nx, Nm)
        )

        for i, idx in enumerate(idxA):

            x = XA[idx]

            for j, (a, w) in enumerate(
                zip(alphas, omegas)
            ):

                PhiA[i, j] = (

                    np.exp(
                        -a*(yI-y0)
                    )

                    *

                    np.sin(
                        w*x
                    )

                )

        for i, idx in enumerate(idxB):

            x = XB[idx]

            for j, (a, w) in enumerate(
                zip(alphas, omegas)
            ):

                PhiB[i, j] = (

                    np.exp(
                        -a*(y1-yI)
                    )

                    *

                    np.sin(
                        w*x
                    )

                )

        I = np.eye(Nx)

        Zxx = np.zeros(
            (Nx, Nx)
        )

        Zxm = np.zeros(
            (Nx, Nm)
        )

        M = np.block([

            [I,   Zxx, Zxm, PhiB],

            [Zxx, I,   PhiA, Zxm],

            [Zxx, Zxx, PhiA, Zxm],

            [Zxx, Zxx, Zxm, PhiB],

        ])

        rhs = []

        for idx in idxA:

            rhs.append(

                interp_previous_A(

                    np.array([XA[idx]]),

                    np.array([y0])

                )[0]

            )

        for idx in idxB:

            rhs.append(

                interp_previous_B(

                    np.array([XB[idx]]),

                    np.array([y1])

                )[0]

            )

        for idx in idxA:

            prev = interp_previous_A(

                np.array([XA[idx]]),

                np.array([y0])

            )[0]

            rhs.append(
                uA[idx] - prev
            )

        for idx in idxB:

            prev = interp_previous_B(

                np.array([XB[idx]]),

                np.array([y1])

            )[0]

            rhs.append(
                uB[idx] - prev
            )

        rhs = np.asarray(rhs)

        lam = 1e-6

        Areg = (
            M.T @ M
            +
            lam*np.eye(
                M.shape[1]
            )
        )

        breg = M.T @ rhs

        x = np.linalg.solve(
            Areg,
            breg,
        )

        traceA = x[:Nx]

        traceB = x[Nx:2*Nx]

        coefA = x[
            2*Nx:
            2*Nx+Nm
        ]

        coefB = x[
            2*Nx+Nm:
        ]

        return (
            coefA,
            coefB,
            traceA,
            traceB,
        )

    @staticmethod
    def correct(basisA,uA,basisB,uB,interp_previous_A,interp_previous_B,geometry):
        if geometry.interface_type == "vertical":

            coefA, coefB, traceA, traceB = (

                InterfaceCorrector.
                _fit_vertical_paper(

                    basisA,
                    uA,

                    basisB,
                    uB,

                    interp_previous_A,
                    interp_previous_B,

                    geometry,
                )
            )

        elif geometry.interface_type == "horizontal":

            coefA, coefB, traceA, traceB = (

                InterfaceCorrector.
                _fit_horizontal_paper(

                    basisA,
                    uA,

                    basisB,
                    uB,

                    interp_previous_A,
                    interp_previous_B,

                    geometry,
                )
            )

        else:

            raise ValueError(
                f"Unsupported interface type "
                f"{geometry.interface_type}"
            )

        XA,YA = basisA.doflocs
        XB,YB = basisB.doflocs

        eA = np.zeros_like(uA)
        eB = np.zeros_like(uB)

        if geometry.interface_type == "vertical":

            for A, alpha, omega in zip(
                coefA,
                geometry.decay_rates,
                geometry.omegas,
            ):

                eA += (

                    A

                    * np.exp(
                        -alpha
                        *
                        (
                            geometry.location
                            - XA
                        )
                    )

                    * np.sin(
                        omega * YA
                    )
                )

            for B, alpha, omega in zip(
                coefB,
                geometry.decay_rates,
                geometry.omegas,
            ):

                eB += (

                    B

                    * np.exp(
                        -alpha
                        *
                        (
                            XB
                            - geometry.location
                        )
                    )

                    * np.sin(
                        omega * YB
                    )
                )

        else:

            for A, alpha, omega in zip(
                coefA,
                geometry.decay_rates,
                geometry.omegas,
            ):

                eA += (

                    A

                    * np.exp(
                        -alpha
                        *
                        (
                            geometry.location
                            - YA
                        )
                    )

                    * np.sin(
                        omega * XA
                    )
                )

            for B, alpha, omega in zip(
                coefB,
                geometry.decay_rates,
                geometry.omegas,
            ):

                eB += (

                    B

                    * np.exp(
                        -alpha
                        *
                        (
                            YB
                            - geometry.location
                        )
                    )

                    * np.sin(
                        omega * XB
                    )
                )

        return CorrectionResult(
            coefficient_A=coefA,
            coefficient_B=coefB,
            correction_A=eA,
            correction_B=eB,
            corrected_A=uA-eA,
            corrected_B=uB-eB,
            trace_A=traceA,
            trace_B=traceB,
        )


@dataclass
class CoupledCorrectionResult:

    trace_x_minus: np.ndarray
    trace_x_plus: np.ndarray

    trace_y_minus: np.ndarray
    trace_y_plus: np.ndarray

    coef_x: np.ndarray
    coef_y: np.ndarray
    coef_xy: np.ndarray

    correction_x: np.ndarray
    correction_y: np.ndarray
    correction_xy: np.ndarray

    correction: np.ndarray

    corrected: np.ndarray


class CoupledInterfaceCorrector:

    @staticmethod
    def fit(
        basis,
        u,
        interp_previous,
        xI,
        yI,
        alphas_x,
        omegas_x,
        alphas_y,
        omegas_y,
        reg=1e-6,
    ):

        X, Y = basis.doflocs

        # ==================================================
        # OBSERVATION LINES
        # ==================================================

        xs = np.unique(X)
        ys = np.unique(Y)

        ix = np.argmin(np.abs(xs - xI))
        iy = np.argmin(np.abs(ys - yI))

        # ----------------------------------
        # protect against interface lying
        # on the domain boundary
        # ----------------------------------

        ix0 = max(ix - 2, 0)
        ix1 = min(ix + 2, len(xs) - 1)

        iy0 = max(iy - 2, 0)
        iy1 = min(iy + 2, len(ys) - 1)

        x0 = xs[ix0]
        x1 = xs[ix1]

        y0 = ys[iy0]
        y1 = ys[iy1]

        idx_x0 = np.where(np.abs(X - x0) < 1e-12)[0]
        idx_x1 = np.where(np.abs(X - x1) < 1e-12)[0]

        idx_y0 = np.where(np.abs(Y - y0) < 1e-12)[0]
        idx_y1 = np.where(np.abs(Y - y1) < 1e-12)[0]

        idx_x0 = idx_x0[np.argsort(Y[idx_x0])]
        idx_x1 = idx_x1[np.argsort(Y[idx_x1])]

        idx_y0 = idx_y0[np.argsort(X[idx_y0])]
        idx_y1 = idx_y1[np.argsort(X[idx_y1])]

        Nxobs = len(idx_x0)
        Nyobs = len(idx_y0)

        Nmx = len(alphas_x)
        Nmy = len(alphas_y)

        # ==================================================
        # VERTICAL MATRICES
        # ==================================================

        Phi_x0 = np.zeros((Nxobs, Nmx))
        Phi_x1 = np.zeros((Nxobs, Nmx))

        for i, idx in enumerate(idx_x0):

            y = Y[idx]

            for j, (a, w) in enumerate(
                zip(alphas_x, omegas_x)
            ):

                Phi_x0[i, j] = (
                    np.exp(
                        -a * (xI - x0)
                    )
                    * np.sin(w * y)
                )

        for i, idx in enumerate(idx_x1):

            y = Y[idx]

            for j, (a, w) in enumerate(
                zip(alphas_x, omegas_x)
            ):

                Phi_x1[i, j] = (
                    np.exp(
                        -a * (x1 - xI)
                    )
                    * np.sin(w * y)
                )

        # ==================================================
        # HORIZONTAL MATRICES
        # ==================================================

        Psi_y0 = np.zeros((Nyobs, Nmy))
        Psi_y1 = np.zeros((Nyobs, Nmy))

        for i, idx in enumerate(idx_y0):

            x = X[idx]

            for j, (a, w) in enumerate(
                zip(alphas_y, omegas_y)
            ):

                Psi_y0[i, j] = (
                    np.exp(
                        -a * (yI - y0)
                    )
                    * np.sin(w * x)
                )

        for i, idx in enumerate(idx_y1):

            x = X[idx]

            for j, (a, w) in enumerate(
                zip(alphas_y, omegas_y)
            ):

                Psi_y1[i, j] = (
                    np.exp(
                        -a * (y1 - yI)
                    )
                    * np.sin(w * x)
                )

        # ==================================================
        # BLOCK MATRICES
        # ==================================================

        Ix = np.eye(Nxobs)
        Iy = np.eye(Nyobs)

        Zxx = np.zeros((Nxobs, Nxobs))
        Zyy = np.zeros((Nyobs, Nyobs))

        Zxmx = np.zeros((Nxobs, Nmx))
        Zxmy = np.zeros((Nxobs, Nmy))

        Zymx = np.zeros((Nyobs, Nmx))
        Zymy = np.zeros((Nyobs, Nmy))

        M = np.block([

            [Ix,  Zxx,
             np.zeros((Nxobs, Nyobs)),
             np.zeros((Nxobs, Nyobs)),
             Zxmx,
             Phi_x1],

            [Zxx, Ix,
             np.zeros((Nxobs, Nyobs)),
             np.zeros((Nxobs, Nyobs)),
             Phi_x0,
             Zxmy],

            [np.zeros((Nyobs, Nxobs)),
             np.zeros((Nyobs, Nxobs)),
             Iy,
             Zyy,
             Zymx,
             Psi_y1],

            [np.zeros((Nyobs, Nxobs)),
             np.zeros((Nyobs, Nxobs)),
             Zyy,
             Iy,
             Psi_y0,
             Zymy],

            [np.zeros((Nxobs,
                        2*Nxobs + 2*Nyobs)),
             Phi_x0,
             Zxmy],

            [np.zeros((Nxobs,
                        2*Nxobs + 2*Nyobs)),
             Phi_x1,
             Zxmy],

            [np.zeros((Nyobs,
                        2*Nxobs + 2*Nyobs)),
             Zymx,
             Psi_y0],

            [np.zeros((Nyobs,
                        2*Nxobs + 2*Nyobs)),
             Zymx,
             Psi_y1],

        ])

        # ==================================================
        # RHS
        # ==================================================

        rhs = []

        for idx in idx_x0:

            rhs.append(
                interp_previous(
                    np.array([x0]),
                    np.array([Y[idx]])
                )[0]
            )

        for idx in idx_x1:

            rhs.append(
                interp_previous(
                    np.array([x1]),
                    np.array([Y[idx]])
                )[0]
            )

        for idx in idx_y0:

            rhs.append(
                interp_previous(
                    np.array([X[idx]]),
                    np.array([y0])
                )[0]
            )

        for idx in idx_y1:

            rhs.append(
                interp_previous(
                    np.array([X[idx]]),
                    np.array([y1])
                )[0]
            )

        for idx in idx_x0:

            prev = interp_previous(
                np.array([x0]),
                np.array([Y[idx]])
            )[0]

            rhs.append(u[idx] - prev)

        for idx in idx_x1:

            prev = interp_previous(
                np.array([x1]),
                np.array([Y[idx]])
            )[0]

            rhs.append(u[idx] - prev)

        for idx in idx_y0:

            prev = interp_previous(
                np.array([X[idx]]),
                np.array([y0])
            )[0]

            rhs.append(u[idx] - prev)

        for idx in idx_y1:

            prev = interp_previous(
                np.array([X[idx]]),
                np.array([y1])
            )[0]

            rhs.append(u[idx] - prev)

        rhs = np.asarray(rhs)

        # ==================================================
        # SOLVE
        # ==================================================

        Areg = (
            M.T @ M
            +
            reg * np.eye(
                M.shape[1]
            )
        )

        z = np.linalg.solve(
            Areg,
            M.T @ rhs,
        )

        # ==================================================
        # UNPACK
        # ==================================================

        p = 0

        tx_minus = z[p:p+Nxobs]
        p += Nxobs

        tx_plus = z[p:p+Nxobs]
        p += Nxobs

        ty_minus = z[p:p+Nyobs]
        p += Nyobs

        ty_plus = z[p:p+Nyobs]
        p += Nyobs

        coef_x = z[p:p+Nmx]
        p += Nmx

        coef_y = z[p:p+Nmy]

        # ==================================================
        # RECONSTRUCTION
        # ==================================================

        ex = np.zeros_like(u)

        for A, alpha, omega in zip(
            coef_x,
            alphas_x,
            omegas_x,
        ):

            ex += (
                A
                *
                np.exp(
                    -alpha
                    * np.abs(X - xI)
                )
                *
                np.sin(
                    omega * Y
                )
            )

        ey = np.zeros_like(u)

        for B, alpha, omega in zip(
            coef_y,
            alphas_y,
            omegas_y,
        ):

            ey += (
                B
                *
                np.exp(
                    -alpha
                    * np.abs(Y - yI)
                )
                *
                np.sin(
                    omega * X
                )
            )

        corr = ex + ey

        return CoupledCorrectionResult(

            trace_x_minus=tx_minus,
            trace_x_plus=tx_plus,

            trace_y_minus=ty_minus,
            trace_y_plus=ty_plus,

            coef_x=coef_x,
            coef_y=coef_y,
            coef_xy=np.array([]),

            correction_x=ex,
            correction_y=ey,
            correction_xy=np.zeros_like(u),

            correction=corr,

            corrected=u - corr,

        )