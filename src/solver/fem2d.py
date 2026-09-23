"""
fem2d.py
========

Finite-element solver for two-dimensional elliptic problems.

Solves

    k*u - div(h grad(u)) = f

on rectangular domains using first-order triangular finite elements.

Features
--------
* Structured rectangular meshes
* Dirichlet boundaries on W, E, N and S
* Support for callable boundary conditions
* Support for tabulated interface conditions
* LinearNDInterpolator construction
* Utilities for overlapping-domain reconstruction

Used by
-------
    elliptic2d.py
    overlap2d.py
    overlapping2d.py
    overlapping2d_4dom.py

Author
------
Lorenzo Zambelli
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.interpolate import LinearNDInterpolator

from skfem import Basis, asm, condense, solve
from skfem.mesh import MeshTri
from skfem.element import ElementTriP1


@dataclass(slots=True)
class FEMSolution2D:

    mesh: MeshTri
    basis: Basis
    solution: np.ndarray
    interpolator: LinearNDInterpolator




def create_rect_mesh(xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int)-> MeshTri:
    """
    Create a triangular mesh for a rectangular domain.

    Inputs:
        xmin, xmax: float
            Domain boundaries in the x-direction.
        ymin, ymax: float
    Outputs:
        mesh: skfem.MeshTri
            The generated triangular mesh.
    """

    return MeshTri.init_tensor(
        np.linspace(xmin,xmax,nx+1),
        np.linspace(ymin,ymax,ny+1),
    )



def boundary_dofs(basis: skfem.Basis, mesh: skfem.MeshTri, side: str) -> np.ndarray:
    """
    Get the degrees of freedom on a specified boundary side of the mesh.

    Inputs:
        basis: skfem.Basis
            The finite element basis on the mesh.
        mesh: skfem.MeshTri
            The triangular mesh.
        side: str
            The boundary side ('W', 'E', 'S', 'N').
    Outputs:
        dofs: np.ndarray
            Array of degrees of freedom on the specified boundary side.
    """

    if side=="W": return basis.get_dofs(lambda x: np.isclose(x[0],mesh.p[0].min()))
    if side=="E": return basis.get_dofs(lambda x: np.isclose(x[0],mesh.p[0].max()))
    if side=="S": return basis.get_dofs(lambda x: np.isclose(x[1],mesh.p[1].min()))
    if side=="N": return basis.get_dofs(lambda x: np.isclose(x[1],mesh.p[1].max()))

    raise ValueError(f"Unknown side {side}")




def assemble_dirichlet(basis: skfem.Basis, mesh: skfem.MeshTri, boundary_conditions: dict) -> tuple:
    """
    Assemble Dirichlet boundary conditions for a 2D finite element problem.

    Inputs:
        basis: skfem.Basis
            The finite element basis on the mesh.
        mesh: skfem.MeshTri
            The triangular mesh.
        boundary_conditions: dict
            Dictionary specifying boundary conditions.
    outputs:
        D: np.ndarray
            Array of Dirichlet degrees of freedom.
        xD: np.ndarray
            Array of prescribed values at Dirichlet degrees of freedom.
    """

    D=[]; prescribed={}; X,Y=basis.doflocs

    for side,bc in boundary_conditions.items():

        ids=np.asarray(boundary_dofs(basis,mesh,side)).ravel()

        D.extend(ids)

        x=X[ids]
        y=Y[ids]

        if callable(bc):

            try:

                values=np.asarray(bc(x,y))

            except TypeError:

                values=np.asarray(
                    bc(y)
                    if side in ("W","E")
                    else bc(x)
                )

        elif isinstance(bc,dict):

            values=np.interp(
                y if side in ("W","E") else x,
                bc["coords"],
                bc["values"],
            )

        else:

            raise ValueError(
                f"Unsupported BC type on {side}"
            )

        if np.ndim(values)==0:

            values=np.full(
                len(ids),
                float(values),
            )

        for idx,value in zip(ids,values):

            prescribed[idx]=float(value)

    D=np.unique(np.asarray(D,dtype=int))

    xD=np.zeros(basis.N,dtype=float)

    for idx,value in prescribed.items():

        xD[idx]=value

    return D,xD


def solve_rect(
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        diffusion_form: callable,
        reaction_form: callable,
        rhs_form: callable,
        boundary_conditions: dict,
    ) -> tuple:
    """
    Solve a 2D elliptic problem on a rectangular domain using finite elements.

    Inputs:
        xmin, xmax: float
            Domain boundaries in the x-direction.
        ymin, ymax: float
            Domain boundaries in the y-direction.
        nx, ny: int
            Number of elements in the x and y directions.
        diffusion_form: callable
            Bilinear form for the diffusion term.
        reaction_form: callable
            Bilinear form for the reaction term.
        rhs_form: callable
            Linear form for the right-hand side.
        boundary_conditions: dict
            Dictionary specifying boundary conditions.
    outputs:
        mesh: skfem.MeshTri
            The generated triangular mesh.
        basis: skfem.Basis
            The finite element basis on the mesh.
        u: np.ndarray
            The solution vector.
    """

    mesh=create_rect_mesh(
        xmin,
        xmax,
        ymin,
        ymax,
        nx,
        ny,
    )

    basis=Basis(
        mesh,
        ElementTriP1(),
    )

    A=asm(diffusion_form,basis)+asm(reaction_form,basis)

    b=asm(rhs_form,basis)

    D,xD=assemble_dirichlet(
        basis,
        mesh,
        boundary_conditions,
    )

    AII,bI,xI,I=condense(A, b,D=D,x=xD,)

    u=xD.copy()

    u[I]=solve(AII,bI)

    return (mesh,basis,u)


def build_interpolator(basis: skfem.Basis, solution: np.ndarray) -> scipy.interpolate.LinearNDInterpolator:
    """
    Build a linear interpolator for the solution on the mesh.
    
    Inputs:
            basis: skfem.Basis
            solution: np.ndarray
    outputs:
        interpolator: scipy.interpolate.LinearNDInterpolator
    """

    return LinearNDInterpolator(
        np.c_[
            basis.doflocs[0],
            basis.doflocs[1],
        ],
        solution,
        fill_value=np.nan,
    )


def solve_rect_interpolated(**kwargs):
    """
    Solve a 2D elliptic problem on a rectangular domain and return an interpolated solution.
    """

    mesh,basis,solution=solve_rect(**kwargs)

    return FEMSolution2D(
        mesh=mesh,
        basis=basis,
        solution=solution,
        interpolator=build_interpolator(
            basis,
            solution,
        ),
    )