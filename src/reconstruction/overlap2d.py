from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


# ============================================================
# TYPES
# ============================================================

Side = Literal["W", "E", "N", "S"]

BasisFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]


# ============================================================
# HOMOGENEOUS EIGENFUNCTIONS
# ============================================================

class EigenFunctions2D:
    """
    Homogeneous eigenfunctions associated with

        -F0 Δu + k u = 0

    on

        [a,b] x [c,d]

    The interface correction is represented as

        u_h = u_left + u_right + u_north + u_south

    where each contribution is expanded in modal form.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        F0: float,
        k: float,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.F0 = F0
        self.k = k

        self.Lx = b - a
        self.Ly = d - c

        self.mu = np.sqrt(k / F0)

    # ---------------------------------------------------------
    # LEFT
    # ---------------------------------------------------------

    def west_homo(
        self,
        n: int,
        x,
        y,
    ):
        beta = n * np.pi / self.Ly
        alpha = np.sqrt(self.mu**2 + beta**2)

        return (
            np.sinh(alpha * (self.b - x))
            / np.sinh(alpha * self.Lx)
            * np.sin(beta * (y - self.c))
        )

    # ---------------------------------------------------------
    # East
    # ---------------------------------------------------------

    def east_homo(
        self,
        n: int,
        x,
        y,
    ):
        beta = n * np.pi / self.Ly
        alpha = np.sqrt(self.mu**2 + beta**2)

        return (
            np.sinh(alpha * (x - self.a))
            / np.sinh(alpha * self.Lx)
            * np.sin(beta * (y - self.c))
        )

    # ---------------------------------------------------------
    # South
    # ---------------------------------------------------------

    def south_homo(
        self,
        n: int,
        x,
        y,
    ):
        gamma = n * np.pi / self.Lx
        alpha = np.sqrt(self.mu**2 + gamma**2)

        return (
            np.sinh(alpha * (self.d - y))
            / np.sinh(alpha * self.Ly)
            * np.sin(gamma * (x - self.a))
        )

    # ---------------------------------------------------------
    # north
    # ---------------------------------------------------------

    def north_homo(
        self,
        n: int,
        x,
        y,
    ):
        gamma = n * np.pi / self.Lx
        alpha = np.sqrt(self.mu**2 + gamma**2)

        return (
            np.sinh(alpha * (y - self.c))
            / np.sinh(alpha * self.Ly)
            * np.sin(gamma * (x - self.a))
        )


# ============================================================
# DOMAIN
# ============================================================

@dataclass(slots=True)
class Domain2D:
    """
    Rectangular subdomain.

    Interfaces indicate which homogeneous families
    participate in the correction.

    Example:

        interfaces=("W","N")

    gives

        u_h = u_west + u_north
    """

    a: float
    b: float
    c: float
    d: float

    F0: float
    k: float

    interfaces: tuple[Side, ...]

    n_modes: int = 10
    weight_function: callable | None = None

    # ---------------------------------------------------------

    @property
    def eigenfunctions(
        self,
    ) -> EigenFunctions2D:

        return EigenFunctions2D(
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            F0=self.F0,
            k=self.k,
        )

    # ---------------------------------------------------------

    def contains(
        self,
        x: float,
        y: float,
    ) -> bool:

        return (
            self.a <= x <= self.b
            and
            self.c <= y <= self.d
        )

    # ---------------------------------------------------------

    @property
    def n_unknowns(
        self,
    ) -> int:

        return (
            len(self.interfaces)
            * self.n_modes
        )

    # ---------------------------------------------------------

    def basis_functions(
        self,
    ) -> tuple[BasisFunction, ...]:
        """
        Returns all modal basis functions for
        the active interfaces.

        Example:

            interfaces=("W","E","N","S")

        returns

            [
                phi_w_1,
                ...
                phi_w_N,

                phi_e_1,
                ...
                phi_e_N
            ]
        """

        ef = self.eigenfunctions

        basis: list[BasisFunction] = []

        for side in self.interfaces:

            if side == "W":

                for n in range(
                    1,
                    self.n_modes + 1,
                ):

                    basis.append(
                        lambda x,
                        y,
                        n=n,
                        ef=ef:
                        ef.west_homo(
                            n,
                            x,
                            y,
                        )
                    )

            elif side == "E":

                for n in range(
                    1,
                    self.n_modes + 1,
                ):

                    basis.append(
                        lambda x,
                        y,
                        n=n,
                        ef=ef:
                        ef.east_homo(
                            n,
                            x,
                            y,
                        )
                    )

            elif side == "N":

                for n in range(
                    1,
                    self.n_modes + 1,
                ):

                    basis.append(
                        lambda x,
                        y,
                        n=n,
                        ef=ef:
                        ef.north_homo(
                            n,
                            x,
                            y,
                        )
                    )

            elif side == "S":

                for n in range(
                    1,
                    self.n_modes + 1,
                ):

                    basis.append(
                        lambda x,
                        y,
                        n=n,
                        ef=ef:
                        ef.south_homo(
                            n,
                            x,
                            y,
                        )
                    )

        return tuple(basis)
    
    def interface_weight(self,side,x,y):

        if side=="W":
            return (self.b-x)/(self.b-self.a)

        elif side=="E":
            return (x-self.a)/(self.b-self.a)

        elif side=="S":
            return (self.d-y)/(self.d-self.c)

        elif side=="N":
            return (y-self.c)/(self.d-self.c)

        raise ValueError(side)


# ============================================================
# SINGLE DOMAIN SOLUTION
# ============================================================

@dataclass(slots=True)
class DomainSolution2D:
    """
    One PDE solution.

    The correction is

        u_corrected = u - u_h

    with

        u_h = Σ a_i φ_i
    """

    domain: Domain2D

    solution: NDArray[np.float64]

    x: NDArray[np.float64]
    y: NDArray[np.float64]

    interpolator: Callable[
        [float, float],
        float,
    ]

    # ---------------------------------------------------------

    @property
    def basis(
        self,
    ) -> tuple[BasisFunction, ...]:

        return (
            self.domain
            .basis_functions()
        )

    # ---------------------------------------------------------

    @property
    def n_unknowns(
        self,
    ) -> int:

        return (
            self.domain
            .n_unknowns
        )


# ============================================================
# COLLECTION OF DOMAINS
# ============================================================

@dataclass(slots=True)
class SolutionDomain2D:
    """
    Collection of overlapping subdomains.

    Linear system:

        u_j(x,y) - u_i(x,y)
            =
        u_j^h(x,y) - u_i^h(x,y)

    for every correction point.
    """

    domains: tuple[
        DomainSolution2D,
        ...
    ]

    # ---------------------------------------------------------

    @property
    def total_unknowns(
        self,
    ) -> int:

        return sum(
            domain.n_unknowns
            for domain in self.domains
        )


# ============================================================
# CORRECTION RESULT
# ============================================================

@dataclass(slots=True)
class OversetCorrectionResult2D:

    corrections: tuple[NDArray[np.float64], ...]

    corrected: tuple[NDArray[np.float64], ...]

    coefficients: NDArray[np.float64]

    full_x: NDArray[np.float64]

    full_y: NDArray[np.float64]

    full_solution: NDArray[np.float64]

    residual: NDArray[np.float64]

    condition_number: float

    residual_norm: float

    normalized_residual: float


# ============================================================
# CORRECTOR SKELETON
# ============================================================

from scipy.linalg import lstsq
from scipy.interpolate import LinearNDInterpolator

import numpy as np


class OversetInterfaceCorrector2D:

    @staticmethod
    def correct(
        U: SolutionDomain2D,
        constraint_points: NDArray[np.float64],
    ) -> OversetCorrectionResult2D:

        # =====================================================
        # GLOBAL UNKNOWN NUMBERING
        # =====================================================

        offsets = []

        offset = 0

        for domain in U.domains:

            offsets.append(offset)

            offset += domain.n_unknowns

        n_unknowns = offset

        # =====================================================
        # BUILD A alpha = b
        # =====================================================

        rows = []
        rhs = []

        for x, y in constraint_points:

            active = []

            for idx, domain in enumerate(U.domains):

                if not domain.domain.contains(x, y):
                    continue

                try:

                    value = float(
                        domain.interpolator(
                            x,
                            y,
                        )
                    )

                except Exception:
                    continue

                if np.isnan(value):
                    continue

                active.append(
                    (
                        idx,
                        domain,
                        value,
                    )
                )

            if len(active) < 2:
                continue

            # ---------------------------------------------
            # Pairwise continuity equations
            # ---------------------------------------------

            for a in range(len(active)):

                for b in range(a + 1, len(active)):

                    idx_i, Di, ui = active[a]
                    idx_j, Dj, uj = active[b]

                    row = np.zeros(
                        n_unknowns,
                        dtype=float,
                    )

                    # ---------------------------------
                    # Domain i contribution
                    # ---------------------------------

                    start_i = offsets[idx_i]

                    for k, phi in enumerate(
                        Di.basis
                    ):

                        row[
                            start_i + k
                        ] -= phi(
                            x,
                            y,
                        )

                    # ---------------------------------
                    # Domain j contribution
                    # ---------------------------------

                    start_j = offsets[idx_j]

                    for k, phi in enumerate(
                        Dj.basis
                    ):

                        row[
                            start_j + k
                        ] += phi(
                            x,
                            y,
                        )

                    rows.append(row)

                    rhs.append(
                        uj - ui
                    )

        if len(rows) == 0:

            raise ValueError(
                "No overlap equations were generated."
            )

        A = np.asarray(rows)

        b = np.asarray(rhs)

        # =====================================================
        # REGULARIZED LEAST SQUARES
        # =====================================================


        reg = 1e-3

        Areg = np.vstack(
            (
                A,
                reg * np.eye(n_unknowns),
            )
        )

        breg = np.concatenate(
            (
                b,
                np.zeros(n_unknowns),
            )
        )

        coefficients, *_ = lstsq(
            Areg,
            breg,
        )

        residual = (
            b
            -
            A @ coefficients
        )

        residual_norm = np.linalg.norm(
            residual
        )

        rhs_norm = np.linalg.norm(
            b
        )

        if rhs_norm > 0:

            normalized_residual = (
                residual_norm
                /
                rhs_norm
            )

        else:

            normalized_residual = np.nan

        condition_number = np.linalg.cond(
            A
        )



        # =====================================================
        # RECONSTRUCT HOMOGENEOUS FIELDS
        # =====================================================

        corrections = []

        corrected = []

        corrected_interpolators = []

        for dom_idx, domain in enumerate(
            U.domains
        ):

            start = offsets[dom_idx]

            end = (
                start
                +
                domain.n_unknowns
            )

            coeffs_domain = (
                coefficients[start:end]
            )

            X = domain.x
            Y = domain.y

            uh = np.zeros_like(
                domain.solution
            )

            for coeff, phi in zip(
                coeffs_domain,
                domain.basis,
            ):

                uh += (
                    coeff
                    *
                    phi(
                        X,
                        Y,
                    )
                )

            uc = (
                domain.solution
                - uh
            )

            corrections.append(uh)

            corrected.append(uc)

            corrected_interpolators.append(
                LinearNDInterpolator(
                    np.c_[X, Y],
                    uc,
                    fill_value=np.nan,
                )
            )

        # =====================================================
        # BUILD GLOBAL SOLUTION
        # =====================================================

        global_points = []

        for domain in U.domains:

            global_points.append(
                np.c_[domain.x, domain.y]
            )

        global_points = np.vstack(
            global_points
        )

        global_points = np.unique(
            global_points,
            axis=0,
        )

        Xg = global_points[:, 0]
        Yg = global_points[:, 1]

        full_solution = np.zeros(
            len(global_points)
        )

        for i, (x, y) in enumerate(
            global_points
        ):

            value_sum = 0.0
            weight_sum = 0.0

            for dom, interp in zip(
                U.domains,
                corrected_interpolators,
            ):

                if not dom.domain.contains(
                    x,
                    y,
                ):
                    continue

                value = interp(
                    x,
                    y,
                )

                if np.isnan(value):
                    continue

                w = dom.domain.weight_function(
                    x,
                    y,
                )

                value_sum += w * float(value)

                weight_sum += w

            if weight_sum == 0:

                full_solution[i] = np.nan

            else:

                full_solution[i] = (
                    value_sum
                    /
                    weight_sum
                )

        return OversetCorrectionResult2D(

            corrections=tuple(corrections),

            corrected=tuple(corrected),

            coefficients=coefficients,

            full_x=Xg,

            full_y=Yg,

            full_solution=full_solution,

            residual=residual,

            condition_number=condition_number,

            residual_norm=residual_norm,

            normalized_residual=normalized_residual,
        )


    @staticmethod
    def weight_west_east(
        x,
        xW,
        xE,
    ):
        return (
            xE - x
        ) / (
            xE - xW
        )

    @staticmethod
    def blend(
        weight,
        uA,
        uB,
    ):
        return (
            weight*uA
            +
            (1-weight)*uB
        )


class ObservationStrategy:

    @staticmethod
    def two_lines(
        xW,
        xE,
        ymin,
        ymax,
        n_y,
    ):

        y = np.linspace(
            ymin,
            ymax,
            n_y,
        )

        west = np.column_stack(
            [
                np.full_like(y, xW),
                y,
            ]
        )

        east = np.column_stack(
            [
                np.full_like(y, xE),
                y,
            ]
        )

        return np.vstack(
            [
                west,
                east,
            ]
        )

    @staticmethod
    def fps(
        candidate_points,
        n_points,
    ):

        return farthest_point_sampling(
            candidate_points,
            n_points,
        )

def farthest_point_sampling(
    points,
    n_select,
):

    points = np.asarray(points)

    if n_select >= len(points):
        return points

    selected = [0]

    distances = np.full(
        len(points),
        np.inf,
    )

    for _ in range(n_select - 1):

        current = points[selected[-1]]

        d = np.linalg.norm(
            points - current,
            axis=1,
        )

        distances = np.minimum(
            distances,
            d,
        )

        selected.append(
            np.argmax(distances)
        )

    return points[selected]