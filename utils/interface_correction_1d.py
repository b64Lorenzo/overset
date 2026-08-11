#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np


@dataclass
class ReconstructionResult:

    coefficients_A: np.ndarray
    coefficients_B: np.ndarray

    reconstructed_A: np.ndarray
    reconstructed_B: np.ndarray

    matrix: np.ndarray
    rhs: np.ndarray

    rank: int
    condition_number: float


class InterfaceReconstructor1D:

    @staticmethod
    def solve(
        xA_obs,
        xB_obs,
        uA_prev,
        uB_prev,
        uA_cur,
        uB_cur,
        modes_A,
        modes_B,
        single_amplitude=False,
        weights_A=None,
        weights_B=None,
    ):

        xA_obs = np.asarray(xA_obs)
        xB_obs = np.asarray(xB_obs)

        uA_prev = np.asarray(uA_prev)
        uB_prev = np.asarray(uB_prev)

        uA_cur = np.asarray(uA_cur)
        uB_cur = np.asarray(uB_cur)

        NA = len(xA_obs)
        NB = len(xB_obs)

        mA = len(modes_A)
        mB = len(modes_B)

        if single_amplitude:

            mA_unknown = 1
            mB_unknown = 1

            if weights_A is None:
                weights_A = np.ones(mA)

            if weights_B is None:
                weights_B = np.ones(mB)

        else:

            mA_unknown = mA
            mB_unknown = mB

        n_unknowns = (
            NA + NB +
            mA_unknown + mB_unknown
        )

        n_equations = 2 * (NA + NB)

        if n_unknowns != n_equations:

            raise ValueError(
                f"System not square:\n"
                f"equations={n_equations}\n"
                f"unknowns={n_unknowns}"
            )

        M = np.zeros(
            (n_equations, n_unknowns),
            dtype=np.complex128
        )

        rhs = np.zeros(
            n_equations,
            dtype=np.complex128
        )

        off_uA = 0
        off_uB = NA

        off_aA = NA + NB
        off_aB = off_aA + mA_unknown

        row = 0

        #
        # reconstruction A
        #

        for k, x in enumerate(xA_obs):

            M[row, off_uA + k] = 1.0

            if single_amplitude:

                phi = 0.0

                for w, (xref, lam) in zip(
                    weights_B,
                    modes_B
                ):
                    phi += (
                        w
                        *
                        np.exp(
                            -lam * (x - xref)
                        )
                    )

                M[row, off_aB] = phi

            else:

                for j, (xref, lam) in enumerate(
                    modes_B
                ):
                    M[row, off_aB + j] = (
                        np.exp(
                            -lam * (x - xref)
                        )
                    )

            rhs[row] = uA_prev[k]
            row += 1

        #
        # reconstruction B
        #

        for k, x in enumerate(xB_obs):

            M[row, off_uB + k] = 1.0

            if single_amplitude:

                phi = 0.0

                for w, (xref, lam) in zip(
                    weights_A,
                    modes_A
                ):
                    phi += (
                        w
                        *
                        np.exp(
                            lam * (x - xref)
                        )
                    )

                M[row, off_aA] = phi

            else:

                for j, (xref, lam) in enumerate(
                    modes_A
                ):
                    M[row, off_aA + j] = (
                        np.exp(
                            lam * (x - xref)
                        )
                    )

            rhs[row] = uB_prev[k]
            row += 1

        #
        # observation A
        #

        for k, x in enumerate(xA_obs):

            if single_amplitude:

                phi = 0.0

                for w, (xref, lam) in zip(
                    weights_A,
                    modes_A
                ):
                    phi += (
                        w
                        *
                        np.exp(
                            lam * (x - xref)
                        )
                    )

                M[row, off_aA] = phi

            else:

                for j, (xref, lam) in enumerate(
                    modes_A
                ):
                    M[row, off_aA + j] = (
                        np.exp(
                            lam * (x - xref)
                        )
                    )

            rhs[row] = (
                uA_cur[k]
                -
                uA_prev[k]
            )

            row += 1

        #
        # observation B
        #

        for k, x in enumerate(xB_obs):

            if single_amplitude:

                phi = 0.0

                for w, (xref, lam) in zip(
                    weights_B,
                    modes_B
                ):
                    phi += (
                        w
                        *
                        np.exp(
                            -lam * (x - xref)
                        )
                    )

                M[row, off_aB] = phi

            else:

                for j, (xref, lam) in enumerate(
                    modes_B
                ):
                    M[row, off_aB + j] = (
                        np.exp(
                            -lam * (x - xref)
                        )
                    )

            rhs[row] = (
                uB_cur[k]
                -
                uB_prev[k]
            )

            row += 1

        rank = np.linalg.matrix_rank(M)

        try:
            cond = np.linalg.cond(M)
        except Exception:
            cond = np.inf

        try:

            sol = np.linalg.solve(M, rhs)

        except np.linalg.LinAlgError:

            sol, *_ = np.linalg.lstsq(
                M,
                rhs,
                rcond=None
            )

        return ReconstructionResult(
            coefficients_A=sol[off_aA:off_aB],
            coefficients_B=sol[off_aB:],
            reconstructed_A=sol[off_uA:off_uB],
            reconstructed_B=sol[off_uB:off_aA],
            matrix=M,
            rhs=rhs,
            rank=rank,
            condition_number=cond,
        )