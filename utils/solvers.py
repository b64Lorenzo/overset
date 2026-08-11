"""
solvers.py
==========

Finite-element solvers for rectangular domains using scikit-fem.

This module provides a generic solver for steady-state reaction-diffusion
problems on rectangular subdomains. The implementation is designed for
domain-decomposition methods (e.g. Schwarz iterations) and supports both
physical boundaries and interface boundaries supplied by neighbouring
subdomains.

Mathematical Problem
--------------------
The solver computes the finite-element approximation of

    -∇ · (D ∇u) + R(u) = f

on a rectangular domain

    Ω = [xmin, xmax] × [ymin, ymax]

subject to Dirichlet boundary conditions.

The variational forms defining the diffusion, reaction, and source terms
must be provided by the caller.

Features
--------
* Structured triangular mesh generation.
* Continuous linear (P1) finite elements.
* Arbitrary physical boundary conditions.
* Multiple interface boundaries.
* Suitable for domain decomposition and Schwarz methods.
* No hard-coded geometry or boundary information.
"""

from __future__ import annotations

import numpy as np

from skfem import (
    MeshTri,
    Basis,
    ElementTriP1,
    asm,
    condense,
    solve,
)


def compute_separation_modes(
    Ly,
    k_reaction,
    F,
    n_modes=10,
):
    """
    Compute separated homogeneous modes

        u(x,y)=v(x)w(y)

    satisfying

        F Δu + k_reaction u = 0

    Parameters
    ----------
    Ly : float
        Domain height.

    k_reaction : float
        Reaction coefficient.

    F : float
        Diffusion coefficient.

    n_modes : int
        Number of modes.

    Returns
    -------
    modes : list[dict]
    """

    lambda_k = k_reaction / F

    modes = []

    for n in range(1, n_modes + 1):

        omega_y = n * np.pi / Ly

        lam = omega_y**2 - lambda_k

        if lam > 0:

            omega_x = np.sqrt(lam)

            v_type = (
                "A exp(omega_x x)"
                " + B exp(-omega_x x)"
            )

        elif lam < 0:

            omega_x = np.sqrt(-lam)

            v_type = (
                "A cos(omega_x x)"
                " + B sin(omega_x x)"
            )

        else:

            omega_x = 0.0

            v_type = "A + B x"

        modes.append(
            {
                "mode": n,
                "lambda": lam,
                "omega_x": omega_x,
                "omega_y": omega_y,
                "v_type": v_type,
            }
        )

    return modes


def solve_rect(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    diffusion_form,
    reaction_form,
    rhs_form,
    boundary_conditions: dict,
):
    """
    Solve a reaction-diffusion problem on a rectangular domain.

    A structured triangular mesh is created over the rectangle

        [xmin, xmax] × [ymin, ymax]

    and a continuous piecewise-linear finite-element approximation
    is used. The system matrix is assembled from the supplied
    variational forms and solved subject to Dirichlet boundary
    conditions.

    This routine is particularly intended for domain-decomposition
    applications. Interface values transferred from neighbouring
    subdomains can be applied on any combination of the domain
    boundaries.

    Parameters
    ----------
    xmin : float
        Minimum x-coordinate of the rectangular domain.

    xmax : float
        Maximum x-coordinate of the rectangular domain.

    ymin : float
        Minimum y-coordinate of the rectangular domain.

    ymax : float
        Maximum y-coordinate of the rectangular domain.

    nx : int
        Number of mesh intervals in the x-direction.

    ny : int
        Number of mesh intervals in the y-direction.

    diffusion_form :
        scikit-fem bilinear form representing the diffusion operator.

    reaction_form :
        scikit-fem bilinear form representing the reaction operator.

    rhs_form :
        scikit-fem linear form defining the source term.

    boundary_conditions : dict
        Dictionary specifying Dirichlet conditions on boundaries.

        Valid boundary identifiers are

        * ``"left"``
        * ``"right"``
        * ``"bottom"``
        * ``"top"``

        Each entry may be one of the following:

        1. Callable boundary condition

           A function with signature

           ``bc(x, y)``

           returning the Dirichlet value at the supplied coordinates.

           Example
           -------
           >>> bc_left = lambda x, y: np.sin(np.pi * y)

        2. Interface condition

           A dictionary containing

           ``coords`` :
               Interface coordinate locations.

           ``values`` :
               Solution values received from a neighbouring
               subdomain.

           Example
           -------
           >>> {
           >>>     "coords": y_interface,
           >>>     "values": interface_solution
           >>> }

           For left/right boundaries interpolation is performed
           along the y-direction.

           For top/bottom boundaries interpolation is performed
           along the x-direction.

    Returns
    -------
    mesh : skfem.MeshTri
        Triangular computational mesh.

    basis : skfem.Basis
        Finite-element basis associated with the mesh.

    u : numpy.ndarray
        Nodal solution vector.

    Notes
    -----
    Boundary conditions are enforced strongly using the
    ``condense`` routine provided by scikit-fem.

    Interface values are transferred using one-dimensional
    linear interpolation via ``numpy.interp``.

    Examples
    --------
    Physical boundary only:

    >>> bc = {
    >>>     "left": bc_left,
    >>>     "right": bc_right,
    >>>     "bottom": bc_bottom,
    >>>     "top": bc_top,
    >>> }

    Domain decomposition:

    >>> bc = {
    >>>     "left": {
    >>>         "coords": y_interface,
    >>>         "values": u_left_interface,
    >>>     },
    >>>     "right": {
    >>>         "coords": y_interface,
    >>>         "values": u_right_interface,
    >>>     },
    >>>     "bottom": bc_bottom,
    >>>     "top": bc_top,
    >>> }
    """

    # ==============================================================
    # Mesh generation
    # ==============================================================

    mesh = MeshTri.init_tensor(
        np.linspace(xmin, xmax, nx + 1),
        np.linspace(ymin, ymax, ny + 1),
    )

    basis = Basis(mesh, ElementTriP1())

    # ==============================================================
    # Assembly of finite-element system
    # ==============================================================

    A = (
        asm(diffusion_form, basis)
        + asm(reaction_form, basis)
    )

    b = asm(rhs_form, basis)

    X = basis.doflocs[0]
    Y = basis.doflocs[1]

    # ==============================================================
    # Boundary degree-of-freedom identification
    # ==============================================================

    tol = 1.0e-12

    boundaries = {
        "left": basis.get_dofs(
            lambda x: np.isclose(x[0], xmin, atol=tol)
        ),
        "right": basis.get_dofs(
            lambda x: np.isclose(x[0], xmax, atol=tol)
        ),
        "bottom": basis.get_dofs(
            lambda x: np.isclose(x[1], ymin, atol=tol)
        ),
        "top": basis.get_dofs(
            lambda x: np.isclose(x[1], ymax, atol=tol)
        ),
    }

    # ==============================================================
    # Collect all constrained degrees of freedom
    # ==============================================================

    constrained_dofs = []

    for side in boundary_conditions:

        if side not in boundaries:
            raise ValueError(
                f"Unknown boundary '{side}'. "
                f"Expected one of "
                f"{list(boundaries.keys())}."
            )

        constrained_dofs.extend(
            np.asarray(boundaries[side]).ravel()
        )

    D = np.unique(constrained_dofs)

    # ==============================================================
    # Construct Dirichlet value vector
    # ==============================================================

    xbc = basis.zeros()

    for side, bc in boundary_conditions.items():

        dofs = np.asarray(boundaries[side]).ravel()

        # ----------------------------------------------------------
        # Standard physical boundary condition
        # ----------------------------------------------------------

        if callable(bc):

            for d in dofs:
                xbc[d] = bc(X[d], Y[d])

        # ----------------------------------------------------------
        # Interface boundary condition
        # ----------------------------------------------------------

        elif isinstance(bc, dict):

            coords = np.asarray(bc["coords"])
            values = np.asarray(bc["values"])

            if side in ("left", "right"):

                for d in dofs:
                    xbc[d] = np.interp(
                        Y[d],
                        coords,
                        values,
                    )

            elif side in ("bottom", "top"):

                for d in dofs:
                    xbc[d] = np.interp(
                        X[d],
                        coords,
                        values,
                    )

        else:

            raise TypeError(
                f"Unsupported boundary condition "
                f"type for boundary '{side}'."
            )

    # ==============================================================
    # Apply Dirichlet constraints
    # ==============================================================

    AII, bI, xI, I = condense(
        A,
        b,
        D=D,
        x=xbc,
    )

    # ==============================================================
    # Solve reduced system
    # ==============================================================

    u = xI.copy()
    u[I] = solve(AII, bI)

    return mesh, basis, u