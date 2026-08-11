#!/usr/bin/env python3
"""
2D overset interface correction.

This is the direct 2D analogue of the user's 1D overset
reconstruction method.

Given:

    xI_left
    xI_right

and modal basis

    phi_n^L(x,y)
    =
    exp(alpha_n (x-xI_left))
    sin(omega_n y)

    phi_n^R(x,y)
    =
    exp(-alpha_n (x-xI_right))
    sin(omega_n y)

the coefficients are reconstructed from overlap traces using
a block linear system exactly analogous to the 1D system

    [u1]
    [u2]
    [a ]
    [b ]

but with modal amplitudes replacing scalar amplitudes.

The overlap interval is

    [xI_right , xI_left]

and the correction decays away from this overlap.

Variable F(x,y) is handled outside the class through the
modal decay rates

    alpha_n
    =
    sqrt(
        omega_n²
        +
        k/F_interface
    )

where

    F_interface

is a suitable average of F evaluated near the interface.
"""

from dataclasses import dataclass
import numpy as np


# ============================================================
# MODES
# ============================================================

def compute_modes(
    Ly,
    k_reaction,
    F_interface,
    n_modes,
):
    """
    Compute interface modes.

    Parameters
    ----------
    Ly : float
        Domain height.

    k_reaction : float

    F_interface : float
        Average diffusion coefficient
        at interface.

    n_modes : int

    Returns
    -------
    alphas : ndarray

    omegas : ndarray
    """

    omegas = (
        np.arange(
            1,
            n_modes + 1
        )
        *
        np.pi
        /
        Ly
    )

    alphas = np.sqrt(
        omegas**2
        +
        k_reaction / F_interface
    )

    return alphas, omegas



# ============================================================
# CORNER MODES
# ============================================================

def compute_corner_modes(
    Lx,
    Ly,
    k_reaction,
    F_interface,
    n_modes_x,
    n_modes_y,
):
    """
    Tensor-product corner modes

        sin(nx*pi*x/Lx)
        sin(ny*pi*y/Ly)

    for

        -F Δu + k u = 0

    with decay

        gamma =
        sqrt(
            omega_x^2
            +
            omega_y^2
            +
            k/F
        )
    """

    gammas = []
    omegas_x = []
    omegas_y = []

    for nx in range(
        1,
        n_modes_x + 1,
    ):

        omega_x = (
            nx
            *
            np.pi
            /
            Lx
        )

        for ny in range(
            1,
            n_modes_y + 1,
        ):

            omega_y = (
                ny
                *
                np.pi
                /
                Ly
            )

            gamma = np.sqrt(

                omega_x**2
                +
                omega_y**2
                +
                k_reaction
                /
                F_interface

            )

            gammas.append(
                gamma
            )

            omegas_x.append(
                omega_x
            )

            omegas_y.append(
                omega_y
            )

    return (

        np.asarray(
            gammas
        ),

        np.asarray(
            omegas_x
        ),

        np.asarray(
            omegas_y
        ),

    )
# ============================================================
# RESULT
# ============================================================

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


# ============================================================
# CORRECTOR
# ============================================================
def trace_at_x(X, Y, U, x_target, y_target):

    xs = np.unique(X)

    i = np.argmin(
        np.abs(xs - x_target)
    )

    x_mesh = xs[i]

    idx = np.where(
        np.isclose(
            X,
            x_mesh,
        )
    )[0]

    idx = idx[
        np.argsort(
            Y[idx]
        )
    ]

    return np.interp(
        y_target,
        Y[idx],
        U[idx],
    )
    
    
def trace_at_y(
    X,
    Y,
    U,
    y_target,
    x_target,
):

    ys = np.unique(Y)

    j = np.argmin(
        np.abs(
            ys - y_target
        )
    )

    y_mesh = ys[j]

    idx = np.where(
        np.isclose(
            Y,
            y_mesh,
        )
    )[0]

    idx = idx[
        np.argsort(
            X[idx]
        )
    ]

    return np.interp(
        x_target,
        X[idx],
        U[idx],
    )
    
    
def left_mode(alpha, k, z):
    return np.exp(alpha*z) * (
        1.0 + 0.1*np.cos(k*z)
    )
    #return np.exp(alpha*z)
    
def right_mode(alpha, k, z):
    return np.exp(-alpha*z) * (
        1.0 + 0.1*np.cos(k*z)
    )
    #return np.exp(-alpha*z)


    
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
        direction="x",
    ):

        XL, YL = basisL.doflocs
        XR, YR = basisR.doflocs
        Lov = interface_left - interface_right

        kx = np.arange(len(alphas)) * np.pi / Lov
        if direction == "x":

            coordL = XL
            coordR = XR

            transL = YL
            transR = YR

        elif direction == "y":

            coordL = YL
            coordR = YR

            transL = XL
            transR = XR

        else:

            raise ValueError(
                "direction must be 'x' or 'y'"
            )

        Nm = len(alphas)

        # =========================================
        # INTERFACE TRACES
        # =========================================

        idxL = np.where(
            np.isclose(
                coordL,
                interface_left,
            )
        )[0]

        idxR = np.where(
            np.isclose(
                coordR,
                interface_right,
            )
        )[0]

        idxL = idxL[
            np.argsort(
                transL[idxL]
            )
        ]

        idxR = idxR[
            np.argsort(
                transR[idxR]
            )
        ]

        sL = transL[idxL]
        sR = transR[idxR]

        Ns = len(idxL)

        uLl = uL[idxL]
        uRr = uR[idxR]

        uRl = np.interp(
            sL,
            sR,
            uRr,
        )

        uLr = np.interp(
            sR,
            sL,
            uLl,
        )

        # =========================================
        # OBSERVATION LOCATIONS
        # =========================================

        Nobs = Nm

        coordsL = np.unique(coordL)
        coordsR = np.unique(coordR)

        obs_left = coordsL[
            (coordsL <= interface_right)
            &
            (coordsL >= interface_left)
        ]

        if len(obs_left):

            obs_left = obs_left[
                np.linspace(
                    0,
                    len(obs_left)-1,
                    min(
                        Nobs,
                        len(obs_left),
                    ),
                    dtype=int,
                )
            ]

        obs_right = coordsR[
            (coordsR <= interface_right)
            &
            (coordsR >= interface_left)
        ]

        if len(obs_right):

            obs_right = obs_right[
                np.linspace(
                    0,
                    len(obs_right)-1,
                    min(
                        Nobs,
                        len(obs_right),
                    ),
                    dtype=int,
                )
            ]

        # =========================================
        # BUILD SYSTEM
        # =========================================

        I = np.eye(Ns)

        Zss = np.zeros(
            (Ns, Ns)
        )

        Zsm = np.zeros(
            (Ns, Nm)
        )

        rows = []
        rhs_list = []

        # =========================================
        # RIGHT DOMAIN OBSERVATIONS
        # =========================================

        for sk in obs_left:

            PhiR = np.zeros(
                (
                    Ns,
                    Nm,
                )
            )

            for j, (
                alpha,
                omega,
            ) in enumerate(
                zip(
                    alphas,
                    omegas,
                )
            ):

                PhiR[:, j] = (
                    right_mode(
                        alpha,
                        kx[j],
                        sk - interface_right,
                    )
                    *
                    np.sin(omega * sL)
                )

            rows.append(
                np.block([
                    I,
                    Zss,
                    Zsm,
                    PhiR,
                ])
            )

            if direction == "x":

                uRk = trace_at_x(
                    XR,
                    YR,
                    uR,
                    sk,
                    sL,
                )

            else:

                uRk = trace_at_y(
                    XR,
                    YR,
                    uR,
                    sk,
                    sL,
                )

            rhs_list.append(
                uRk
            )

        # =========================================
        # LEFT DOMAIN OBSERVATIONS
        # =========================================

        for sk in obs_right:

            PhiL = np.zeros(
                (
                    Ns,
                    Nm,
                )
            )

            for j, (
                alpha,
                omega,
            ) in enumerate(
                zip(
                    alphas,
                    omegas,
                )
            ):

                PhiL[:, j] = (
                    left_mode(
                        alpha,
                        kx[j],
                        sk - interface_left,
                    )
                    *
                    np.sin(omega * sR)
                )

            rows.append(
                np.block([
                    Zss,
                    I,
                    PhiL,
                    Zsm,
                ])
            )

            if direction == "x":

                uLk = trace_at_x(
                    XL,
                    YL,
                    uL,
                    sk,
                    sR,
                )

            else:

                uLk = trace_at_y(
                    XL,
                    YL,
                    uL,
                    sk,
                    sR,
                )

            rhs_list.append(
                uLk
            )

        # =========================================
        # INTERFACE EQUATIONS
        # =========================================

        PhiL_bc = np.zeros(
            (
                Ns,
                Nm,
            )
        )

        PhiR_bc = np.zeros(
            (
                Ns,
                Nm,
            )
        )

        for j, omega in enumerate(
            omegas
        ):

            PhiL_bc[:, j] = np.sin(
                omega * sL
            )

            PhiR_bc[:, j] = np.sin(
                omega * sR
            )

        rows.append(
            np.block([
                I,
                Zss,
                PhiL_bc,
                Zsm,
            ])
        )

        rhs_list.append(
            uLl
        )

        rows.append(
            np.block([
                Zss,
                I,
                Zsm,
                PhiR_bc,
            ])
        )

        rhs_list.append(
            uRr
        )

        # =========================================
        # SOLVE
        # =========================================

        M = np.vstack(
            rows
        )

        rhs = np.concatenate(
            rhs_list
        )

        sol, *_ = np.linalg.lstsq(
            M,
            rhs,
            rcond=None,
        )

        # =========================================
        # UNPACK
        # =========================================

        p = 0

        trace_left = sol[
            p:p+Ns
        ]
        p += Ns

        trace_right = sol[
            p:p+Ns
        ]
        p += Ns

        coef_left = sol[
            p:p+Nm
        ]
        p += Nm

        coef_right = sol[
            p:p+Nm
        ]

        # =========================================
        # RECONSTRUCTION
        # =========================================

        correction_left = np.zeros_like(
            uL
        )

        correction_right = np.zeros_like(
            uR
        )

        if direction == "x":

            for j, (c, alpha, omega) in enumerate(
                    zip(
                        coef_left,
                        alphas,
                        omegas,
                    )
                ):


                correction_left += (

                    c

                    *

                    left_mode(
                        alpha,
                        kx[j],
                        XL - interface_left
                    )


                    *

                    np.sin(
                        omega
                        *
                        YL
                    )

                )

            for j, (c, alpha, omega) in enumerate(
                zip(
                    coef_right,
                    alphas,
                    omegas,
                )
            ):

                correction_right += (

                    c

                    *

                    right_mode(
                        alpha,
                        kx[j],
                        XR - interface_right
                    )

                    *

                    np.sin(
                        omega
                        *
                        YR
                    )

                )

        else:

            for j, (c, alpha, omega) in enumerate(
                zip(
                    coef_left,
                    alphas,
                    omegas,
                )
            ):

                correction_left += (

                    c

                    *

                    left_mode(
                        alpha,
                        kx[j],
                        YL - interface_left
                    )

                    *

                    np.sin(
                        omega * XL
                    )

                )


            for j, (c, alpha, omega) in enumerate(
                zip(
                    coef_right,
                    alphas,
                    omegas,
                )
            ):

                correction_right += (

                    c

                    *

                    right_mode(
                        alpha,
                        kx[j],
                        YR - interface_right
                    )

                    *

                    np.sin(
                        omega * XR
                    )

                )

        corrected_left = (
            uL
            -
            correction_left
        )

        corrected_right = (
            uR
            -
            correction_right
        )

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
        
        
# ============================================================
# GEOMETRY
# ============================================================

from dataclasses import dataclass
import numpy as np

def trace_at_y(
    X,
    Y,
    U,
    y_target,
    x_target,
):

    ys = np.unique(Y)

    j = np.argmin(
        np.abs(
            ys - y_target
        )
    )

    y_mesh = ys[j]

    idx = np.where(
        np.isclose(
            Y,
            y_mesh,
        )
    )[0]

    idx = idx[
        np.argsort(
            X[idx]
        )
    ]

    return np.interp(
        x_target,
        X[idx],
        U[idx],
    )



def box_coordinates(
    X,
    Y,
    geometry,
):

    xi = (
        X
        -
        geometry.x_overlap_right
    ) / (
        geometry.x_overlap_left
        -
        geometry.x_overlap_right
    )

    eta = (
        Y
        -
        geometry.y_overlap_top
    ) / (
        geometry.y_overlap_bottom
        -
        geometry.y_overlap_top
    )

    return xi, eta

@dataclass
class OversetCrossCorrectionResult2D:

    trace_x: np.ndarray
    trace_y: np.ndarray

    coefficient_x: np.ndarray
    coefficient_y: np.ndarray
    coefficient_xy: np.ndarray

    correction_x: np.ndarray
    correction_y: np.ndarray
    correction_xy: np.ndarray

    correction: np.ndarray
    corrected: np.ndarray

@dataclass
class OversetCrossGeometry2D:

    x_interface: float
    y_interface: float

    x_sign: float
    y_sign: float

    x_overlap_left: float
    x_overlap_right: float

    y_overlap_top: float
    y_overlap_bottom: float

    alphas_x: np.ndarray
    omegas_x: np.ndarray

    alphas_y: np.ndarray
    omegas_y: np.ndarray

    gammas_xy: np.ndarray

    omegas_xy_x: np.ndarray
    omegas_xy_y: np.ndarray

class OversetCrossInterfaceCorrector2D:

    @staticmethod
    def correct(
        basis,
        u,
        interp_x_neighbor,
        interp_y_neighbor,
        geometry,
    ):

        X, Y = basis.doflocs

        NmX = len(geometry.alphas_x)
        NmY = len(geometry.alphas_y)
        NmXY = len(geometry.gammas_xy)

        # --------------------------------------------
        # X-interface trace
        # --------------------------------------------

        xs = np.unique(X)

        xmesh = xs[
            np.argmin(
                np.abs(
                    xs
                    -
                    geometry.x_interface
                )
            )
        ]

        idxX = np.where(
            np.isclose(
                X,
                xmesh,
            )
        )[0]

        idxX = idxX[
            np.argsort(
                Y[idxX]
            )
        ]

        yTrace = Y[idxX]

        # --------------------------------------------
        # Y-interface trace
        # --------------------------------------------

        ys = np.unique(Y)

        ymesh = ys[
            np.argmin(
                np.abs(
                    ys
                    -
                    geometry.y_interface
                )
            )
        ]

        idxY = np.where(
            np.isclose(
                Y,
                ymesh,
            )
        )[0]

        idxY = idxY[
            np.argsort(
                X[idxY]
            )
        ]

        xTrace = X[idxY]

        Ny = len(yTrace)
        Nx = len(xTrace)

        # --------------------------------------------
        # Interface data
        # --------------------------------------------

        trace_x_data = u[idxX]
        trace_y_data = u[idxY]

        # --------------------------------------------
        # Overlap observations
        # --------------------------------------------

        xobs = xs[
            (xs >= geometry.x_overlap_right)
            &
            (xs <= geometry.x_overlap_left)
        ]

        yobs = ys[
            (ys >= geometry.y_overlap_top)
            &
            (ys <= geometry.y_overlap_bottom)
        ]

        if len(xobs):

            xobs = xobs[
                np.linspace(
                    0,
                    len(xobs)-1,
                    min(
                        NmX,
                        len(xobs),
                    ),
                    dtype=int,
                )
            ]

        if len(yobs):

            yobs = yobs[
                np.linspace(
                    0,
                    len(yobs)-1,
                    min(
                        NmY,
                        len(yobs),
                    ),
                    dtype=int,
                )
            ]

        # --------------------------------------------
        # Unknown count
        # --------------------------------------------

        n_unknowns = (

            Ny

            +

            Nx

            +

            NmX

            +

            NmY

            +

            NmXY

        )

        # --------------------------------------------
        # Block matrices
        # --------------------------------------------

        Ix = np.eye(Ny)
        Iy = np.eye(Nx)

        Zxy = np.zeros((Ny, Nx))
        Zyx = np.zeros((Nx, Ny))

        ZX = np.zeros((Nx, NmX))
        ZY = np.zeros((Ny, NmY))

        ZXYy = np.zeros((Ny, NmXY))
        ZXYx = np.zeros((Nx, NmXY))

        rows = []
        rhs = []

        # ==================================================
        # Vertical overlap equations
        #
        # neighbor
        #
        # =
        #
        # trace_x
        #
        # + X modes
        #
        # + XY modes
        # ==================================================

        for xk in xobs:

            PhiX = np.zeros(
                (
                    Ny,
                    NmX,
                )
            )

            PhiXY = np.zeros(
                (
                    Ny,
                    NmXY,
                )
            )

            for j, (
                alpha,
                omega,
            ) in enumerate(
                zip(
                    geometry.alphas_x,
                    geometry.omegas_x,
                )
            ):

                PhiX[:, j] = (

                    np.exp(

                        geometry.x_sign
                        *
                        alpha
                        *
                        (
                            xk
                            -
                            geometry.x_interface
                        )

                    )

                    *

                    np.sin(
                        omega
                        *
                        yTrace
                    )

                )

            dx = (
                xk
                -
                geometry.x_interface
            )

            dy = (
                yTrace
                -
                geometry.y_interface
            )

            for k, (
                gamma,
                ox,
                oy,
            ) in enumerate(
                zip(
                    geometry.gammas_xy,
                    geometry.omegas_xy_x,
                    geometry.omegas_xy_y,
                )
            ):

                PhiXY[:, k] = (

                    np.exp(

                        gamma

                        *

                        (

                            geometry.x_sign
                            * dx

                            +

                            geometry.y_sign
                            * dy

                        )

                    )

                    *

                    np.sin(
                        ox * dx
                    )

                    *

                    np.sin(
                        oy * dy
                    )

                )

            rows.append(

                np.hstack([

                    Ix,
                    Zxy,

                    PhiX,

                    ZY,

                    PhiXY,

                ])

            )

            rhs.append(

                interp_x_neighbor(
                    np.full_like(
                        yTrace,
                        xk,
                    ),
                    yTrace,
                )

            )

        # ==================================================
        # Horizontal overlap equations
        # ==================================================

        for yk in yobs:

            PhiY = np.zeros(
                (
                    Nx,
                    NmY,
                )
            )

            PhiXY = np.zeros(
                (
                    Nx,
                    NmXY,
                )
            )

            for j, (
                alpha,
                omega,
            ) in enumerate(
                zip(
                    geometry.alphas_y,
                    geometry.omegas_y,
                )
            ):

                PhiY[:, j] = (

                    np.exp(

                        geometry.y_sign
                        *
                        alpha
                        *
                        (
                            yk
                            -
                            geometry.y_interface
                        )

                    )

                    *

                    np.sin(
                        omega
                        *
                        xTrace
                    )

                )

            dx = (
                xTrace
                -
                geometry.x_interface
            )

            dy = (
                yk
                -
                geometry.y_interface
            )

            for k, (
                gamma,
                ox,
                oy,
            ) in enumerate(
                zip(
                    geometry.gammas_xy,
                    geometry.omegas_xy_x,
                    geometry.omegas_xy_y,
                )
            ):

                PhiXY[:, k] = (

                    np.exp(

                        gamma

                        *

                        (

                            geometry.x_sign
                            * dx

                            +

                            geometry.y_sign
                            * dy

                        )

                    )

                    *

                    np.sin(
                        ox * dx
                    )

                    *

                    np.sin(
                        oy * dy
                    )

                )

            rows.append(

                np.hstack([

                    Zyx,
                    Iy,

                    ZX,

                    PhiY,

                    PhiXY,

                ])

            )

            rhs.append(

                interp_y_neighbor(
                    xTrace,
                    np.full_like(
                        xTrace,
                        yk,
                    ),
                )

            )

        # ==================================================
        # Vertical interface equation
        #
        # u(xI,y)
        #
        # =
        #
        # trace_x
        #
        # + X modes
        # ==================================================

        PhiXbc = np.zeros(
            (
                Ny,
                NmX,
            )
        )

        for j, omega in enumerate(
            geometry.omegas_x
        ):

            PhiXbc[:, j] = np.sin(
                omega
                *
                yTrace
            )

        rows.append(

            np.hstack([

                Ix,
                Zxy,

                PhiXbc,

                ZY,

                ZXYy,

            ])

        )

        rhs.append(
            trace_x_data
        )

        # ==================================================
        # Horizontal interface equation
        # ==================================================

        PhiYbc = np.zeros(
            (
                Nx,
                NmY,
            )
        )

        for j, omega in enumerate(
            geometry.omegas_y
        ):

            PhiYbc[:, j] = np.sin(
                omega
                *
                xTrace
            )

        rows.append(

            np.hstack([

                Zyx,
                Iy,

                ZX,

                PhiYbc,

                ZXYx,

            ])

        )

        rhs.append(
            trace_y_data
        )

        # ==================================================
        # Crossing compatibility
        # ==================================================

        center_y = np.argmin(
            np.abs(
                yTrace
                -
                geometry.y_interface
            )
        )

        center_x = np.argmin(
            np.abs(
                xTrace
                -
                geometry.x_interface
            )
        )

        eq = np.zeros(
            n_unknowns
        )

        eq[center_y] = 1.0

        eq[
            Ny
            +
            center_x
        ] = -1.0

        rows.append(
            eq[None, :]
        )

        rhs.append(
            np.array([0.0])
        )

        # ==================================================
        # Solve
        # ==================================================

        M = np.vstack(rows)

        rhs = np.concatenate(rhs)

        sol, *_ = np.linalg.lstsq(
            M,
            rhs,
            rcond=None,
        )

        # ==================================================
        # Unpack
        # ==================================================

        p = 0

        trace_x = sol[p:p+Ny]
        p += Ny

        trace_y = sol[p:p+Nx]
        p += Nx

        coef_x = sol[p:p+NmX]
        p += NmX

        coef_y = sol[p:p+NmY]
        p += NmY

        coef_xy = sol[p:p+NmXY]



        # =====================================
        # RECONSTRUCTION
        # =====================================

        correction_x = np.zeros_like(
            u
        )

        for c, alpha, omega in zip(
            coef_x,
            geometry.alphas_x,
            geometry.omegas_x,
        ):

            correction_x += (

                c

                *

                np.exp(
                    geometry.x_sign
                    *
                    alpha
                    *
                    (
                        X
                        -
                        geometry.x_interface
                    )
                )

                *

                np.sin(
                    omega * Y
                )

            )

        correction_y = np.zeros_like(
            u
        )

        for c, alpha, omega in zip(
            coef_y,
            geometry.alphas_y,
            geometry.omegas_y,
        ):

            correction_y += (

                c

                *

                np.exp(
                    geometry.y_sign
                    *
                    alpha
                    *
                    (
                        Y
                        -
                        geometry.y_interface
                    )
                )

                *

                np.sin(
                    omega * X
                )

            )

        correction_xy = np.zeros_like(
            u
        )

        dx = (
            X
            -
            geometry.x_interface
        )

        dy = (
            Y
            -
            geometry.y_interface
        )

        for c, gamma, ox, oy in zip(

            coef_xy,

            geometry.gammas_xy,

            geometry.omegas_xy_x,
            geometry.omegas_xy_y,

        ):

            correction_xy += (

                c

                *

                np.exp(

                    gamma

                    *

                    (

                        geometry.x_sign
                        * dx

                        +

                        geometry.y_sign
                        * dy

                    )

                )

                *

                np.sin(
                    ox * dx
                )

                *

                np.sin(
                    oy * dy
                )

            )

        correction = (
            correction_x
            +
            correction_y
            +
            correction_xy
        )

        corrected = (
            u
            -
            correction
        )

        return OversetCrossCorrectionResult2D(

            trace_x=trace_x,
            trace_y=trace_y,

            coefficient_x=coef_x,
            coefficient_y=coef_y,
            coefficient_xy=coef_xy,

            correction_x=correction_x,
            correction_y=correction_y,
            correction_xy=correction_xy,

            correction=correction,

            corrected=corrected,
        )