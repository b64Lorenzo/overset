"""
utils.py
========

Utility functions used by the one-dimensional
domain-decomposition framework.

The routines provided here are intentionally
independent of the finite-element solver and are
reused throughout:

* overlap reconstruction
* flux reconstruction
* interface matching
* validation studies

The functions provide:

* one-sided derivative approximations
* interface flux evaluation
* global solution reconstruction

Author
------
Lorenzo Zambelli
"""

from __future__ import annotations

import numpy as np


def derivative_east(
    u: np.ndarray,
    dx: float,
) -> float:
    """
    Compute the spatial derivative at the east
    boundary of a one-dimensional grid using a
    second-order accurate one-sided stencil.

    The approximation is

        du/dx ≈
        ( 3u_N
        - 4u_{N-1}
        + u_{N-2} )
        / (2Δx)

    Parameters
    ----------
    u : ndarray
        Solution values.

    dx : float
        Grid spacing.

    Returns
    -------
    float
        Approximation of the derivative at the
        eastern boundary.

    Examples
    --------
    >>> derivative_east(u, dx)

    Used when evaluating the flux leaving the
    western subdomain at an interface.
    """

    return (
        3.0 * u[-1]
        - 4.0 * u[-2]
        + u[-3]
    ) / (2.0 * dx)


def derivative_west(
    u: np.ndarray,
    dx: float,
) -> float:
    """
    Compute the spatial derivative at the west
    boundary of a one-dimensional grid using a
    second-order accurate one-sided stencil.

    The approximation is

        du/dx ≈
        (-3u_0
        +4u_1
        -u_2)
        /(2Δx)

    Parameters
    ----------
    u : ndarray
        Solution values.

    dx : float
        Grid spacing.

    Returns
    -------
    float
        Approximation of the derivative at the
        western boundary.

    Examples
    --------
    >>> derivative_west(u, dx)

    Used when evaluating the flux entering the
    eastern subdomain from an interface.
    """

    return (
        -3.0 * u[0]
        + 4.0 * u[1]
        - u[2]
    ) / (2.0 * dx)


def interface_flux(
    u_west: np.ndarray,
    u_east: np.ndarray,
    dx_west: float,
    dx_east: float,
    x_interface: float,
    diffusion_coefficient: callable,
) -> tuple[float, float]:
    """
    Evaluate the diffusive flux on both sides
    of an interface.

    The diffusive flux is

        q = h(x) u_x

    and is computed using second-order one-sided
    derivative approximations on each subdomain.

    Parameters
    ----------
    u_west : ndarray
        Solution on the western subdomain.

    u_east : ndarray
        Solution on the eastern subdomain.

    dx_west : float
        Grid spacing in the western subdomain.

    dx_east : float
        Grid spacing in the eastern subdomain.

    x_interface : float
        Interface location.

    diffusion_coefficient : callable
        Function returning h(x).

    Returns
    -------
    tuple
        (flux_west, flux_east)

    Notes
    -----
    The returned values correspond to

        flux_west = h(xI) u_x^west

        flux_east = h(xI) u_x^east

    evaluated at the interface.

    Examples
    --------
    >>> q_west, q_east = interface_flux(
    ...     u_west,
    ...     u_east,
    ...     dx_west,
    ...     dx_east,
    ...     x_interface,
    ...     F_func,
    ... )
    """

    coefficient = diffusion_coefficient(
        x_interface
    )

    flux_west = (
        coefficient
        * derivative_east(
            u_west,
            dx_west,
        )
    )

    flux_east = (
        coefficient
        * derivative_west(
            u_east,
            dx_east,
        )
    )

    return (
        flux_west,
        flux_east,
    )


def reconstruct_solution(
    x_global: np.ndarray,
    x_interface: float,
    x_west: np.ndarray,
    u_west: np.ndarray,
    x_east: np.ndarray,
    u_east: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a global solution from two
    subdomain solutions.

    The western solution is used for

        x <= x_interface

    and the eastern solution is used for

        x > x_interface.

    Parameters
    ----------
    x_global : ndarray
        Coordinates of the global mesh.

    x_interface : float
        Interface location.

    x_west : ndarray
        Coordinates of the western subdomain.

    u_west : ndarray
        Solution on the western subdomain.

    x_east : ndarray
        Coordinates of the eastern subdomain.

    u_east : ndarray
        Solution on the eastern subdomain.

    Returns
    -------
    ndarray
        Reconstructed global solution.

    Examples
    --------
    >>> u_global = reconstruct_solution(
    ...     x_full,
    ...     xI,
    ...     x_west,
    ...     u_west,
    ...     x_east,
    ...     u_east,
    ... )
    """

    reconstructed = np.zeros_like(
        x_global
    )

    west_mask = (
        x_global
        <=
        x_interface
    )

    reconstructed[west_mask] = np.interp(
        x_global[west_mask],
        x_west,
        u_west,
    )

    reconstructed[~west_mask] = np.interp(
        x_global[~west_mask],
        x_east,
        u_east,
    )

    return reconstructed