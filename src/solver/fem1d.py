"""
fem1d.py
========

Author
------
Lorenzo Zambelli

Version
-------
1.0

Description
-----------
Finite-element solver for one-dimensional elliptic boundary-value
problems arising from the overlap-reconstruction framework.

The solver discretizes and solves

    k*u - d/dx(h*u_x)
        =
        d/dx(p*f_x)

using first-order finite elements.

The implementation is intentionally independent of the reconstruction
methodology and can therefore be used for

* benchmark solutions on the full domain,
* local subdomain solutions,
* homogeneous auxiliary problems,
* validation studies.

Mathematical Problem
--------------------
Find

    u : [a,b] -> R

such that

    k*u - d/dx(h*u_x)
        =
        d/dx(p*f_x)

subject to Dirichlet boundary conditions

    u(a) = g_a
    u(b) = g_b.

Weak Form
---------
Find u in V such that

    a(u,v) = l(v)

for all v in V,

with

    a(u,v)
        =
        ∫ h*u_x*v_x dx
        +
        ∫ k*u*v dx

and

    l(v)
        =
        ∫ rhs*v dx.

Dependencies
------------
numpy
scikit-fem

References
----------
Lawrence et al. (2011)

Klopman (2010)

Olver (2014)
"""

from __future__ import annotations

import numpy as np

from skfem import (
    MeshLine,
    Basis,
    ElementLineP1,
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)

from skfem.helpers import grad, dot


class FEMEllipticSolver1D:
    """
    Finite-element solver for one-dimensional elliptic problems.

    Parameters
    ----------
    problem : Elliptic1DProblem
        Physical problem definition.

    Notes
    -----
    The solver does not assume anything about overlap
    reconstruction. Its purpose is to provide numerical
    solutions of the governing PDE on arbitrary intervals.
    """

    def __init__(self, problem):

        self.problem = problem

    # --------------------------------------------------
    # Assembly forms
    # --------------------------------------------------

    def _diffusion_form(self):
        """
        Construct diffusion bilinear form.

        Returns
        -------
        BilinearForm
            Form representing

                ∫ h*u_x*v_x dx.
        """

        problem = self.problem

        @BilinearForm
        def diffusion(u, v, w):
            return (
                problem.h(w.x[0])
                *
                dot(
                    grad(u),
                    grad(v),
                )
            )

        return diffusion

    def _reaction_form(self):
        """
        Construct reaction bilinear form.

        Returns
        -------
        BilinearForm
            Form representing

                ∫ k*u*v dx.
        """

        k = self.problem.k

        @BilinearForm
        def reaction(u, v, w):
            return k * u * v

        return reaction

    def _rhs_form(self, current_time):
        """
        Construct right-hand-side form.

        Parameters
        ----------
        current_time : float
            Evaluation time.

        Returns
        -------
        LinearForm
            Load-vector integrand.
        """

        problem = self.problem

        @LinearForm
        def rhs(v, w):

            return (
                problem.forcing(
                    w.x[0],
                    current_time,
                )
                * v
            )

        return rhs

    # --------------------------------------------------
    # Solver
    # --------------------------------------------------

    def solve(
        self,
        a,
        b,
        n_elements,
        bc_left,
        bc_right,
        current_time=0.0,
        homogeneous=False,
    ):
        """
        Solve the elliptic problem on [a,b].

        Parameters
        ----------
        a : float
            Left boundary.

        b : float
            Right boundary.

        n_elements : int
            Number of finite elements.

        bc_left : float
            Dirichlet condition at x=a.

        bc_right : float
            Dirichlet condition at x=b.

        current_time : float, optional
            Evaluation time.

        homogeneous : bool, optional
            If True solve

                k*u - d/dx(h*u_x) = 0.

            Otherwise solve the complete PDE.

        Returns
        -------
        x : ndarray
            Nodal coordinates.

        u : ndarray
            Nodal solution values.
        """

        mesh = MeshLine(
            np.linspace(
                a,
                b,
                n_elements + 1,
            )
        )

        basis = Basis(
            mesh,
            ElementLineP1(),
        )

        diffusion = asm(
            self._diffusion_form(),
            basis,
        )

        reaction = asm(
            self._reaction_form(),
            basis,
        )

        A = diffusion + reaction

        if homogeneous:

            rhs_vector = np.zeros(
                basis.N
            )

        else:

            rhs_vector = asm(
                self._rhs_form(
                    current_time
                ),
                basis,
            )

        left_dofs = basis.get_dofs(
            lambda x: np.isclose(
                x[0],
                a,
            )
        )

        right_dofs = basis.get_dofs(
            lambda x: np.isclose(
                x[0],
                b,
            )
        )

        D = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        left_dofs
                    ).ravel(),
                    np.asarray(
                        right_dofs
                    ).ravel(),
                ]
            )
        )

        x_bc = basis.zeros()

        for dof in np.asarray(
            left_dofs
        ).ravel():

            x_bc[dof] = bc_left

        for dof in np.asarray(
            right_dofs
        ).ravel():

            x_bc[dof] = bc_right

        AII, bI, xI, I = condense(
            A,
            rhs_vector,
            D=D,
            x=x_bc,
        )

        solution = xI.copy()

        solution[I] = solve(
            AII,
            bI,
        )

        return (
            mesh.p[0],
            solution,
        )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    def residual(
        self,
        x,
        u,
        current_time=0.0,
    ):
        """
        Compute PDE residual.

        Parameters
        ----------
        x : ndarray
            Coordinates.

        u : ndarray
            Numerical solution.

        current_time : float
            Evaluation time.

        Returns
        -------
        lhs : ndarray
            Left-hand side.

        rhs : ndarray
            Right-hand side.

        residual : ndarray
            lhs - rhs.
        """

        dx = np.mean(np.diff(x))

        ux = np.gradient(
            u,
            dx,
            edge_order=2,
        )

        flux = (
            self.problem.h(x)
            * ux
        )

        flux_x = np.gradient(
            flux,
            dx,
            edge_order=2,
        )

        lhs = (-flux_x + self.problem.k * u)

        rhs = self.problem.forcing(x,current_time,)

        return (lhs, rhs,lhs - rhs,)