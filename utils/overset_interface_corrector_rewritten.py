#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np


def compute_modes(Ly, k_reaction, F_interface, n_modes):
    omegas = np.arange(1, n_modes + 1) * np.pi / Ly
    alphas = np.sqrt(omegas**2 + k_reaction / F_interface)
    return alphas, omegas


@dataclass
class OversetCorrectionResult2D:
    trace_left: np.ndarray
    trace_right: np.ndarray
    coefficient_left: np.ndarray
    coefficient_right: np.ndarray
    correction_left: np.ndarray
    correction_right: np.ndarray
    corrected_left: np.ndarray
    corrected_right: np.ndarray


def trace_at_x(X, Y, U, x_target, y_target):
    xs = np.unique(X)
    x_mesh = xs[np.argmin(np.abs(xs - x_target))]
    idx = np.where(np.isclose(X, x_mesh))[0]
    idx = idx[np.argsort(Y[idx])]
    return np.interp(y_target, Y[idx], U[idx])


def trace_at_y(X, Y, U, y_target, x_target):
    ys = np.unique(Y)
    y_mesh = ys[np.argmin(np.abs(ys - y_target))]
    idx = np.where(np.isclose(Y, y_mesh))[0]
    idx = idx[np.argsort(X[idx])]
    return np.interp(x_target, X[idx], U[idx])


def left_basis(alpha, x):
    return np.exp(alpha * x)


def right_basis(alpha, x):
    return np.exp(-alpha * x)


class OversetInterfaceCorrector2D:

    @staticmethod
    def correct(
        basisL,
        uL,
        basisR,
        uR,
        interface_left,
        interface_right,
        interface_center,
        alphas,
        omegas,
        direction='x',
    ):

        XL, YL = basisL.doflocs
        XR, YR = basisR.doflocs

        if direction == 'x':
            coordL = XL
            coordR = XR
            transL = YL
            transR = YR
            sampler = trace_at_x
        elif direction == 'y':
            coordL = YL
            coordR = YR
            transL = XL
            transR = XR
            sampler = trace_at_y
        else:
            raise ValueError("direction must be 'x' or 'y'")

        idxL = np.where(np.isclose(coordL, interface_left))[0]
        idxR = np.where(np.isclose(coordR, interface_right))[0]

        idxL = idxL[np.argsort(transL[idxL])]
        idxR = idxR[np.argsort(transR[idxR])]

        sL = transL[idxL]
        sR = transR[idxR]

        Ns = len(idxL)
        Nm = len(alphas)

        s = sL

        overlap_width = interface_left - interface_right
        dx = overlap_width / (Nm - 1)

        I = np.eye(Ns)
        Zss = np.zeros((Ns, Ns))
        Zsm = np.zeros((Ns, Nm))

        rows = []
        rhs_blocks = []

        for i in range(Nm):

            xL = interface_left - i * dx
            xR = interface_right + i * dx

            PhiL_xL = np.zeros((Ns, Nm))
            PhiL_xR = np.zeros((Ns, Nm))
            PhiR_xL = np.zeros((Ns, Nm))
            PhiR_xR = np.zeros((Ns, Nm))

            for j, (alpha, omega) in enumerate(zip(alphas, omegas)):

                PhiL_xL[:, j] = left_basis(alpha, xL - interface_left) * np.sin(omega * s)
                PhiL_xR[:, j] = left_basis(alpha, xR - interface_left) * np.sin(omega * s)
                PhiR_xL[:, j] = right_basis(alpha, xL - interface_right) * np.sin(omega * s)
                PhiR_xR[:, j] = right_basis(alpha, xR - interface_right) * np.sin(omega * s)

            rows.append(np.block([I,   Zss, Zsm,     PhiR_xL]))
            rhs_blocks.append(sampler(XR, YR, uR, xL, s))

            rows.append(np.block([Zss, I,   PhiL_xR, Zsm]))
            rhs_blocks.append(sampler(XL, YL, uL, xR, s))

            rows.append(np.block([I,   Zss, PhiL_xL, Zsm]))
            rhs_blocks.append(sampler(XL, YL, uL, xL, s))

            rows.append(np.block([Zss, I,   Zsm,     PhiR_xR]))
            rhs_blocks.append(sampler(XR, YR, uR, xR, s))

        A = np.vstack(rows)
        rhs = np.concatenate(rhs_blocks)

        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        p = 0
        trace_left = sol[p:p+Ns]; p += Ns
        trace_right = sol[p:p+Ns]; p += Ns
        coef_left = sol[p:p+Nm]; p += Nm
        coef_right = sol[p:p+Nm]

        correction_left = np.zeros_like(uL)
        correction_right = np.zeros_like(uR)

        if direction == 'x':
            for c, alpha, omega in zip(coef_left, alphas, omegas):
                correction_left += c * left_basis(alpha, XL - interface_left) * np.sin(omega * YL)

            for c, alpha, omega in zip(coef_right, alphas, omegas):
                correction_right += c * right_basis(alpha, XR - interface_right) * np.sin(omega * YR)

        else:
            for c, alpha, omega in zip(coef_left, alphas, omegas):
                correction_left += c * left_basis(alpha, YL - interface_left) * np.sin(omega * XL)

            for c, alpha, omega in zip(coef_right, alphas, omegas):
                correction_right += c * right_basis(alpha, YR - interface_right) * np.sin(omega * XR)

        corrected_left = uL - correction_left
        corrected_right = uR - correction_right

        return OversetCorrectionResult2D(
            trace_left=trace_left,
            trace_right=trace_right,
            coefficient_left=coef_left,
            coefficient_right=coef_right,
            correction_left=correction_left,
            correction_right=correction_right,
            corrected_left=corrected_left,
            corrected_right=corrected_right,
        )
