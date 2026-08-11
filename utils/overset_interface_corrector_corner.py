#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np


def compute_modes(Ly, k_reaction, F_interface, n_modes):
    omegas = np.arange(1, n_modes + 1) * np.pi / Ly
    alphas = np.sqrt(omegas**2 + k_reaction / F_interface)
    return alphas, omegas


def trace_at_x(X, Y, U, x_target, y_target):
    xs = np.unique(X)
    x_mesh = xs[np.argmin(np.abs(xs - x_target))]
    idx = np.where(np.isclose(X, x_mesh))[0]
    idx = idx[np.argsort(Y[idx])]
    return np.interp(y_target, Y[idx], U[idx])


def left_basis(alpha, x):
    return np.exp(alpha * x)


def right_basis(alpha, x):
    return np.exp(-alpha * x)


@dataclass
class OversetCorrectionResult2D:
    trace_left: np.ndarray
    trace_right: np.ndarray
    coefficient_left: np.ndarray
    coefficient_right: np.ndarray
    coefficient_corner: np.ndarray
    correction_left: np.ndarray
    correction_right: np.ndarray
    correction_corner: np.ndarray
    corrected_left: np.ndarray
    corrected_right: np.ndarray


class OversetInterfaceCorrector2D:

    @staticmethod
    def correct(
        basisL,
        uL,
        basisR,
        uR,
        interface_left,
        interface_right,
        corner_x,
        corner_y,
        alphas,
        omegas,
        betas,
        mus,
                direction='x',
    ):

        XL, YL = basisL.doflocs
        XR, YR = basisR.doflocs

        idxL = np.where(np.isclose(XL, interface_left))[0]
        idxR = np.where(np.isclose(XR, interface_right))[0]

        idxL = idxL[np.argsort(YL[idxL])]
        idxR = idxR[np.argsort(YR[idxR])]

        s = YL[idxL]

        Ns = len(s)
        Nm = len(alphas)

        Ncorner = len(alphas) * len(betas)

        overlap_width = interface_left - interface_right
        dx = overlap_width / (Nm - 1)

        I = np.eye(Ns)
        Zss = np.zeros((Ns, Ns))
        Zm = np.zeros((Ns, Nm))
        Zc = np.zeros((Ns, Ncorner))

        rows = []
        rhs_blocks = []

        for i in range(Nm):

            xL = interface_left - i * dx
            xR = interface_right + i * dx

            PhiL_xL = np.zeros((Ns, Nm))
            PhiL_xR = np.zeros((Ns, Nm))
            PhiR_xL = np.zeros((Ns, Nm))
            PhiR_xR = np.zeros((Ns, Nm))

            for k, (alpha, omega) in enumerate(zip(alphas, omegas)):
                PhiL_xL[:, k] = left_basis(alpha, xL-interface_left) * np.sin(omega*s)
                PhiL_xR[:, k] = left_basis(alpha, xR-interface_left) * np.sin(omega*s)
                PhiR_xL[:, k] = right_basis(alpha, xL-interface_right) * np.sin(omega*s)
                PhiR_xR[:, k] = right_basis(alpha, xR-interface_right) * np.sin(omega*s)

            PhiXY_xL = np.zeros((Ns, Ncorner))
            PhiXY_xR = np.zeros((Ns, Ncorner))

            p = 0
            for ia,(alpha,omega) in enumerate(zip(alphas,omegas)):
                for jb,(beta,mu) in enumerate(zip(betas,mus)):

                    PhiXY_xL[:,p] = (
                        np.exp(-alpha*np.abs(xL-corner_x))
                        * np.exp(-beta*np.abs(s-corner_y))
                        * np.sin(omega*s)
                        * np.sin(mu*xL)
                    )

                    PhiXY_xR[:,p] = (
                        np.exp(-alpha*np.abs(xR-corner_x))
                        * np.exp(-beta*np.abs(s-corner_y))
                        * np.sin(omega*s)
                        * np.sin(mu*xR)
                    )
                    p += 1

            rows.append(np.block([I,Zss,Zm,PhiR_xL,PhiXY_xL]))
            rhs_blocks.append(trace_at_x(XR,YR,uR,xL,s))

            rows.append(np.block([Zss,I,PhiL_xR,Zm,PhiXY_xR]))
            rhs_blocks.append(trace_at_x(XL,YL,uL,xR,s))

            rows.append(np.block([I,Zss,PhiL_xL,Zm,PhiXY_xL]))
            rhs_blocks.append(trace_at_x(XL,YL,uL,xL,s))

            rows.append(np.block([Zss,I,Zm,PhiR_xR,PhiXY_xR]))
            rhs_blocks.append(trace_at_x(XR,YR,uR,xR,s))

        A = np.vstack(rows)
        rhs = np.concatenate(rhs_blocks)

        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        p = 0
        trace_left = sol[p:p+Ns]; p += Ns
        trace_right = sol[p:p+Ns]; p += Ns
        coef_left = sol[p:p+Nm]; p += Nm
        coef_right = sol[p:p+Nm]; p += Nm
        coef_corner = sol[p:p+Ncorner]

        correction_left = np.zeros_like(uL)
        correction_right = np.zeros_like(uR)
        correction_corner_L = np.zeros_like(uL)
        correction_corner_R = np.zeros_like(uR)

        for c, alpha, omega in zip(coef_left, alphas, omegas):
            correction_left += c * left_basis(alpha, XL-interface_left) * np.sin(omega*YL)

        for c, alpha, omega in zip(coef_right, alphas, omegas):
            correction_right += c * right_basis(alpha, XR-interface_right) * np.sin(omega*YR)

        p = 0
        for alpha,omega in zip(alphas,omegas):
            for beta,mu in zip(betas,mus):
                cc = coef_corner[p]

                correction_corner_L += (
                    cc
                    * np.exp(-alpha*np.abs(XL-corner_x))
                    * np.exp(-beta*np.abs(YL-corner_y))
                    * np.sin(omega*YL)
                    * np.sin(mu*XL)
                )

                correction_corner_R += (
                    cc
                    * np.exp(-alpha*np.abs(XR-corner_x))
                    * np.exp(-beta*np.abs(YR-corner_y))
                    * np.sin(omega*YR)
                    * np.sin(mu*XR)
                )
                p += 1

        corrected_left = uL - correction_left - correction_corner_L
        corrected_right = uR - correction_right - correction_corner_R

        return OversetCorrectionResult2D(
            trace_left=trace_left,
            trace_right=trace_right,
            coefficient_left=coef_left,
            coefficient_right=coef_right,
            coefficient_corner=coef_corner,
            correction_left=correction_left,
            correction_right=correction_right,
            correction_corner=correction_corner_L,
            corrected_left=corrected_left,
            corrected_right=corrected_right,
        )
