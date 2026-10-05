#!/usr/bin/env python3

"""
flux2d_validation.py

2D West-East disjoint flux reconstruction validation.

Produces:

    error_before_2d.png
    error_after_2d.png
    error_comparison_2d_flux.png

    interface_slice.png

    homogeneous_correction_flux2d.png

    west_homogeneous_error.png
    east_homogeneous_error.png

    full_validation_results.csv

Validation metrics:

    L2 before reconstruction
    L2 after reconstruction
    Improvement factor

    West homogeneous error
    East homogeneous error

    Cond(A)
"""

from src.utils.logger import Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import LinearNDInterpolator

from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad

from src.solver.fem2d import solve_rect

from src.reconstruction.flux_2d import (
    Domain2D,
    DomainSolution2D,
    FluxInterfaceCorrector2D,
)

# ==========================================================
# LOGGING
# ==========================================================

plt.rcParams.update(
    {
        "font.size": 14
    }
)


log_mgr = Logger(
    name="flux2d",
    run_prefix="flux2d",
)

logger = log_mgr.get_logger()

run_dir = log_mgr.get_run_dir()

# ==========================================================
# DOMAIN
# ==========================================================

Lx = 100.0
Ly = 40.0

xI = 64.0

# ==========================================================
# MESH
# ==========================================================

nx_full = 300
ny_full = 200

nx_W = 200
nx_E = 200
n_constraints = 4
n_modes = 2
# ==========================================================
# PDE
# ==========================================================

F = 30.0
k_reaction = 25.0

# ==========================================================
# FORCING
# ==========================================================

@BilinearForm
def diffusion(u,v,w):
    return F * dot(
        grad(u),
        grad(v)
    )

@BilinearForm
def reaction(u,v,w):
    return k_reaction*u*v

@LinearForm
def rhs(v,w):
    return 0.0*v

# ==========================================================
# BC
# ==========================================================

full_bc = {

    "W":
    lambda x,y: 0.0,

    "E":
    lambda x,y: 0.0,

    "S":
    lambda x,y: 0.0,

    "N":
    lambda x,y: 0.0,
}

# ==========================================================
# REFERENCE
# ==========================================================

logger.info(
    "Solving reference solution"
)

meshF,basisF,uF = solve_rect(

    xmin=0.0,
    xmax=Lx,

    ymin=0.0,
    ymax=Ly,

    nx=nx_full,
    ny=ny_full,

    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,

    boundary_conditions=full_bc,
)

interp_full = LinearNDInterpolator(
    np.c_[
        basisF.doflocs[0],
        basisF.doflocs[1]
    ],
    uF
)


# ==========================================================
# FINE INTERFACE GRID
# used for BC interpolation
# ==========================================================

y_interface_fine = np.linspace(
    0.0,
    Ly,
    ny_full + 1
)

# ==========================================================
# RECONSTRUCTION GRID
# used for solving for homogeneous amplitudes
# ==========================================================

y_interface_rec = np.linspace(
    0.0,
    Ly,
    n_constraints
)

uI_exact = interp_full(
    np.full_like(
        y_interface_fine,
        xI
    ),
    y_interface_fine
)

amplitude = 10.0

uW_interface = (
    uI_exact
    +
    amplitude
    *
    np.sin(
        2*np.pi*y_interface_fine/Ly
    )
)

uE_interface = (
    uI_exact
    -
    amplitude
    *
    np.sin(
        2*np.pi*y_interface_fine/Ly
    )
)

# ==========================================================
# WEST DOMAIN
# ==========================================================

logger.info(
    "Solving West domain"
)

bcW = {

    "W":
    lambda x,y: 0.0,

    "S":
    lambda x,y: 0.0,

    "N":
    lambda x,y: 0.0,

    "E": {
        "coords": y_interface_fine,
        "values": uW_interface,
    },
}

meshW,basisW,uW = solve_rect(

    xmin=0.0,
    xmax=xI,

    ymin=0.0,
    ymax=Ly,

    nx=nx_W,
    ny=ny_full,

    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,

    boundary_conditions=bcW,
)

# ==========================================================
# EAST DOMAIN
# ==========================================================

logger.info(
    "Solving East domain"
)

bcE = {

    "W": {
        "coords": y_interface_fine,
        "values": uE_interface,
    },

    "E":
    lambda x,y: 0.0,

    "S":
    lambda x,y: 0.0,

    "N":
    lambda x,y: 0.0,
}

meshE,basisE,uE = solve_rect(

    xmin=xI,
    xmax=Lx,

    ymin=0.0,
    ymax=Ly,

    nx=nx_E,
    ny=ny_full,

    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,

    boundary_conditions=bcE,
)

# ==========================================================
# INTERPOLATORS
# ==========================================================

interpW = LinearNDInterpolator(
    np.c_[basisW.doflocs[0],
          basisW.doflocs[1]],
    uW
)

interpE = LinearNDInterpolator(
    np.c_[basisE.doflocs[0],
          basisE.doflocs[1]],
    uE
)

# ==========================================================
# GRADIENTS
# ==========================================================

def build_gradx(X,Y,U):

    interp = LinearNDInterpolator(
        np.c_[X,Y],
        U
    )

    def gradx(x,y):

        h = 1e-6

        return (
            interp(x+h,y)
            -
            interp(x-h,y)
        )/(2*h)

    return gradx

# ==========================================================
# DOMAIN OBJECTS
# ==========================================================

domainW = DomainSolution2D(

    domain=Domain2D(
        0.0,
        xI,
        0.0,
        Ly,
        F,
        k_reaction,
        interfaces=("e",),
        n_modes=n_modes,
    ),

    solution=uW,

    x=basisW.doflocs[0],
    y=basisW.doflocs[1],

    interpolator=interpW,

    gradx=build_gradx(
        basisW.doflocs[0],
        basisW.doflocs[1],
        uW,
    ),

    grady=None,
)

domainE = DomainSolution2D(

    domain=Domain2D(
        xI,
        Lx,
        0.0,
        Ly,
        F,
        k_reaction,
        interfaces=("w",),
        n_modes=n_modes,
    ),

    solution=uE,

    x=basisE.doflocs[0],
    y=basisE.doflocs[1],

    interpolator=interpE,

    gradx=build_gradx(
        basisE.doflocs[0],
        basisE.doflocs[1],
        uE,
    ),

    grady=None,
)

# ==========================================================
# RECONSTRUCTION
# ==========================================================

logger.info(
    "Running flux reconstruction"
)

result = (
    FluxInterfaceCorrector2D
    .correct(
        domains=(
            domainW,
            domainE,
        ),
        y_interface=y_interface_rec,
        x_interface=xI,
    )
)

condA = np.linalg.cond(
    result.matrix
)

logger.info(
    f"Cond(A) = {condA:.6e}"
)

eW,eE = result.corrections

uWc,uEc = result.corrected

# ==========================================================
# TRUE HOMOGENEOUS
# ==========================================================

XW,YW = basisW.doflocs
XE,YE = basisE.doflocs

uW_ref = interp_full(XW,YW)
uE_ref = interp_full(XE,YE)

uWh = uW-uW_ref
uEh = uE-uE_ref

hom_err_W = uWh-eW
hom_err_E = uEh-eE

L2_hom_W = np.linalg.norm(
    hom_err_W
)

L2_hom_E = np.linalg.norm(
    hom_err_E
)

logger.info(
    f"L2 hom West = {L2_hom_W:.6e}"
)

logger.info(
    f"L2 hom East = {L2_hom_E:.6e}"
)

# ==========================================================
# GLOBAL FIELD
# ==========================================================

Xf = basisF.doflocs[0]
Yf = basisF.doflocs[1]

u_unc = np.zeros_like(uF)
u_rec = np.zeros_like(uF)

mask = Xf <= xI

u_unc[mask] = interpW(
    Xf[mask],
    Yf[mask]
)

u_unc[~mask] = interpE(
    Xf[~mask],
    Yf[~mask]
)

interpWc = LinearNDInterpolator(
    np.c_[XW,YW],
    uWc
)

interpEc = LinearNDInterpolator(
    np.c_[XE,YE],
    uEc
)

u_rec[mask] = interpWc(
    Xf[mask],
    Yf[mask]
)

u_rec[~mask] = interpEc(
    Xf[~mask],
    Yf[~mask]
)

err_before = u_unc-uF
err_after = u_rec-uF

L2_before = np.linalg.norm(
    err_before
)

L2_after = np.linalg.norm(
    err_after
)

logger.info(
    f"L2 before = {L2_before:.6e}"
)

logger.info(
    f"L2 after = {L2_after:.6e}"
)

logger.info(
    f"Improvement = {L2_before/L2_after:.3f}"
)
# ==========================================================
# UNKNOWNS / CONSTRAINTS
# ==========================================================

nmodes = domainW.domain.n_modes

n_unknowns = 2 * nmodes

n_solution_constraints = len(
    y_interface_rec
)

n_flux_constraints = len(
    y_interface_rec
)

n_constraints = (
    n_solution_constraints
    +
    n_flux_constraints
)

logger.info(
    f"Unknown amplitudes = {n_unknowns}"
)

logger.info(
    f"Solution constraints = {n_solution_constraints}"
)

logger.info(
    f"Flux constraints = {n_flux_constraints}"
)

logger.info(
    f"Total constraints = {n_constraints}"
)

logger.info(
    f"Overdetermined ratio = "
    f"{n_constraints/n_unknowns:.3f}"
)
# ==========================================================
# ERROR BEFORE
# ==========================================================

plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    err_before,
    levels=100,
    cmap="seismic"
)

plt.colorbar()

plt.title(
    "Error Before Reconstruction"
)

plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()

plt.savefig(
    run_dir /
    "error_before_2d.png",
    dpi=400
)

plt.close()
# ==========================================================
# ERROR AFTER
# ==========================================================

plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    err_after,
    levels=100,
    cmap="seismic"
)

plt.colorbar()

plt.title(
    "Error After Reconstruction"
)

plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()

plt.savefig(
    run_dir /
    "error_after_2d.png",
    dpi=400
)

plt.close()
# ==========================================================
# ERROR COMPARISON
# ==========================================================

fig,ax = plt.subplots(
    1,
    2,
    figsize=(14,6)
)

cf1 = ax[0].tricontourf(
    Xf,
    Yf,
    err_before,
    levels=100,
    cmap="seismic"
)

plt.colorbar(
    cf1,
    ax=ax[0]
)

ax[0].set_title(
    "Before Reconstruction"
)

cf2 = ax[1].tricontourf(
    Xf,
    Yf,
    err_after,
    levels=100,
    cmap="seismic"
)

plt.colorbar(
    cf2,
    ax=ax[1]
)

ax[1].set_title(
    "After Reconstruction"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "error_comparison_2d_flux.png",
    dpi=400
)

plt.close()
# ==========================================================
# HOMOGENEOUS VALIDATION
# ==========================================================

fig,(ax1,ax2)=plt.subplots(
    2,
    1,
    figsize=(12,10)
)

mn = min(
    np.min(uWh),
    np.min(eW)
)

mx = max(
    np.max(uWh),
    np.max(eW)
)

ax1.scatter(
    uWh,
    eW,
    s=1
)

ax1.plot(
    [mn,mx],
    [mn,mx],
    'r--'
)

ax1.set_title(
    "West Homogeneous Correction"
)

ax1.set_xlabel(
    "True"
)

ax1.set_ylabel(
    "Reconstructed"
)

ax1.text(
    0.02,
    0.95,
    f"L2={L2_hom_W:.3e}",
    transform=ax1.transAxes,
)

mn = min(
    np.min(uEh),
    np.min(eE)
)

mx = max(
    np.max(uEh),
    np.max(eE)
)

ax2.scatter(
    uEh,
    eE,
    s=1
)

ax2.plot(
    [mn,mx],
    [mn,mx],
    'r--'
)

ax2.set_title(
    "East Homogeneous Correction"
)

ax2.set_xlabel(
    "True"
)

ax2.set_ylabel(
    "Reconstructed"
)

ax2.text(
    0.02,
    0.95,
    f"L2={L2_hom_E:.3e}",
    transform=ax2.transAxes,
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "homogeneous_correction_flux2d.png",
    dpi=400
)

plt.close()
print(
    "Validation finished"
)