#!/usr/bin/env python3
"""
1D Validation with configurable interface location and independent meshes.

Key parameters:
    xI      : interface location
    n_left  : elements in left subdomain
    n_right : elements in right subdomain

PDE:
    -(F(x)u_x)_x + k u = (P(x)f_x)_x
"""

import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfem import MeshLine, Basis, ElementLineP1
from skfem import BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad
from src.utils.logger import Logger

plt.rcParams.update({'font.size':14,'axes.titlesize':18,'axes.labelsize':16})

log_mgr = Logger()

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()

# -------------------------------------------------
# USER PARAMETERS
# -------------------------------------------------
Lx = 100.0
xI = 64            # interface can be anywhere in domain
n_full = 800         # benchmark mesh
n_left = 400         # left subdomain resolution
n_right = 400        # right subdomain resolution

k_reaction = 25.0
relax = 0.64
noise_amplitude = 6.4
rng = np.random.default_rng(42)

# -------------------------------------------------
# COEFFICIENTS
# -------------------------------------------------
F0 = 30.0
P0 = 10.0
P_amp = 64
k_p = 0.64

A_f = 1.0
k_f = 0.25
omega = 0.64
current_time = 0.0


def F_func(x):
    return F0 + 5.0*np.cos(0.64*x)
    #return F0 
    
    

def F_prime(x, h=1e-6):
    return (F_func(x + h) - F_func(x - h)) / (2.0 * h)
    #return 0

def P_func(x):
    return P0 + P_amp*np.cos(k_p*x)


def P_prime(x):
    return -P_amp*k_p*np.sin(k_p*x)


def f_x(x,t):
    return A_f*k_f*np.cos(k_f*x-omega*t)
    #return  0
    


def f_xx(x,t):
    return -A_f*(k_f**2)*np.sin(k_f*x-omega*t)
    #return 0


def forcing_term(x):
    return P_prime(x)*f_x(x,current_time) + P_func(x)*f_xx(x,current_time)


def left_bc(x):
    #return 1.5*np.sin(0.1*x)
    return 0


def right_bc(x):
    #return 1.5*np.sin(0.1*x)
    return 0


@BilinearForm
def diffusion(u,v,w):
    return F_func(w.x[0]) * dot(grad(u),grad(v))


@BilinearForm
def reaction(u,v,w):
    return k_reaction*u*v


@LinearForm
def rhs(v,w):
    return forcing_term(w.x[0]) * v



def solve_domain(a,b,n,bc_left,bc_right,homogeneous=False):
    mesh=MeshLine(np.linspace(a,b,n+1))
    basis=Basis(mesh,ElementLineP1())

    A=asm(diffusion,basis)+asm(reaction,basis)
    bvec=np.zeros(basis.N) if homogeneous else asm(rhs,basis)

    dL=basis.get_dofs(lambda x: np.isclose(x[0],a))
    dR=basis.get_dofs(lambda x: np.isclose(x[0],b))

    D=np.unique(np.concatenate([np.asarray(dL).ravel(),np.asarray(dR).ravel()]))

    xbc=basis.zeros()
    for d in np.asarray(dL).ravel(): xbc[d]=bc_left
    for d in np.asarray(dR).ravel(): xbc[d]=bc_right

    AII,bI,xIvec,I=condense(A,bvec,D=D,x=xbc)
    u=xIvec.copy(); u[I]=solve(AII,bI)
    return mesh.p[0],u


def pde_balance(x,u):
    dx=np.mean(np.diff(x))
    ux=np.gradient(u,dx,edge_order=2)
    flux=F_func(x)*ux
    fluxx=np.gradient(flux,dx,edge_order=2)
    lhs=-fluxx+k_reaction*u
    rhsv=forcing_term(x)
    return lhs,rhsv,lhs-rhsv

# BENCHMARK
x_full,u_full=solve_domain(0,Lx,n_full,left_bc(0),right_bc(Lx))
ui_exact=np.interp(xI,x_full,u_full)
ul_guess=ui_exact + noise_amplitude*rng.standard_normal()
ur_guess=ui_exact + noise_amplitude*rng.standard_normal()

# SUBDOMAINS
xL,uL=solve_domain(0,xI,n_left,left_bc(0),ul_guess)
xR,uR=solve_domain(xI,Lx,n_right,ur_guess,right_bc(Lx))
# ============================================================
# INTERFACE CORRECTION
# ============================================================

uLI = uL[-1]
uRI = uR[0]

Ju = uLI - uRI

dxL = xL[-1] - xL[-2]
dxR = xR[1] - xR[0]

duLI = (
    3.0*uL[-1]
    - 4.0*uL[-2]
    + uL[-3]
) / (2.0*dxL)

duRI = (
    -3.0*uR[0]
    + 4.0*uR[1]
    - uR[2]
) / (2.0*dxR)

Fv = F_func(xI)

fluxL = Fv * duLI
fluxR = Fv * duRI

Jflux = fluxL - fluxR

# ------------------------------------------------------------
# HOMOGENEOUS UNIT RESPONSES
# phiL(xI)=1, phiR(xI)=1
# ------------------------------------------------------------

_, phiL = solve_domain(
    0.0,
    xI,
    n_left,
    0.0,
    1.0,
    homogeneous=True
)

_, phiR = solve_domain(
    xI,
    Lx,
    n_right,
    1.0,
    0.0,
    homogeneous=True
)

# ------------------------------------------------------------
# FLUX RESPONSE OF BASIS FUNCTIONS
# ------------------------------------------------------------

qphiL = Fv * (
    3.0*phiL[-1]
    - 4.0*phiL[-2]
    + phiL[-3]
) / (2.0*dxL)

qphiR = Fv * (
    -3.0*phiR[0]
    + 4.0*phiR[1]
    - phiR[2]
) / (2.0*dxR)

# ------------------------------------------------------------
# SOLVE FOR TWO AMPLITUDES
#
# Enforces:
#   uLc(xI) = uRc(xI)
#   fluxLc(xI) = fluxRc(xI)
# ------------------------------------------------------------

A = np.array([
    [1.0,    -1.0],
    [qphiL, -qphiR]
])

rhs = np.array([
    Ju,
    Jflux
])

epsL, epsR = np.linalg.solve(A, rhs)

condA = np.linalg.cond(A)

logger.info(
    "Cond(A)=%.6e",
    condA
)

# ------------------------------------------------------------
# CORRECTION FIELDS
# ------------------------------------------------------------

eL = epsL * phiL
eR = epsR * phiR

uLc = uL - eL
uRc = uR - eR


# ============================================================
# TRUE HOMOGENEOUS CORRECTIONS
# ============================================================

u_exact_left = np.interp(xL, x_full, u_full)
u_exact_right = np.interp(xR, x_full, u_full)

true_homogeneous_left = uL - u_exact_left
true_homogeneous_right = uR - u_exact_right

reconstructed_homogeneous_left = eL
reconstructed_homogeneous_right = eR

left_h_error = np.linalg.norm(
    true_homogeneous_left
    - reconstructed_homogeneous_left
)

right_h_error = np.linalg.norm(
    true_homogeneous_right
    - reconstructed_homogeneous_right
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(13,10)
)

ax1.plot(
    xL,
    true_homogeneous_left,
    lw=3,
    label='True correction'
)

ax1.plot(
    xL,
    reconstructed_homogeneous_left,
    '--',
    lw=3,
    label='Reconstructed correction'
)

ax1.text(
    0.02,
    0.95,
    f"L2={left_h_error:.3e}",
    transform=ax1.transAxes,
    va='top'
)

ax1.grid(True)
ax1.legend()
ax1.set_title(
    "West Homogeneous Correction"
)

ax2.plot(
    xR,
    true_homogeneous_right,
    lw=3,
    label='True correction'
)

ax2.plot(
    xR,
    reconstructed_homogeneous_right,
    '--',
    lw=3,
    label='Reconstructed correction'
)

ax2.text(
    0.02,
    0.95,
    f"L2={right_h_error:.3e}",
    transform=ax2.transAxes,
    va='top'
)

ax2.grid(True)
ax2.legend()
ax2.set_title(
    "East Homogeneous Correction"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "homogeneous_correction_flux.png",
    dpi=600
)

# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------

fluxLc = Fv * (
    3.0*uLc[-1]
    - 4.0*uLc[-2]
    + uLc[-3]
) / (2.0*dxL)

fluxRc = Fv * (
    -3.0*uRc[0]
    + 4.0*uRc[1]
    - uRc[2]
) / (2.0*dxR)

print("Ju before    =", Ju)
print("Jflux before =", Jflux)

print("Ju after     =", uLc[-1] - uRc[0])
print("Jflux after  =", fluxLc - fluxRc)
print("epsL         =", epsL)
print("epsR         =", epsR)
uLh = uL - u_full[:n_left+1]
uRh = uR - u_full[n_left:]



u_unc=np.zeros_like(u_full)
u_rec=np.zeros_like(u_full)
mask=x_full<=xI
u_unc[mask]=np.interp(x_full[mask],xL,uL)
u_unc[~mask]=np.interp(x_full[~mask],xR,uR)
u_rec[mask]=np.interp(x_full[mask],xL,uLc)
u_rec[~mask]=np.interp(x_full[~mask],xR,uRc)

err_before=np.linalg.norm(u_unc-u_full)
err_after=np.linalg.norm(u_rec-u_full)

lhsf,rhsf,resf=pde_balance(x_full,u_full)

err_before=np.linalg.norm(u_unc-u_full)
err_after=np.linalg.norm(u_rec-u_full)

lhsf,rhsf,resf=pde_balance(x_full,u_full)

# ============================================================
# SOLUTION + ERROR PLOT
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(16, 9)
)

# Solutions
ax1.plot(
    x_full,
    u_full,
    color='black',
    lw=4,
    label='Exact Solution'
)

ax1.plot(
    x_full,
    u_unc,
    '--',
    color='red',
    lw=2,
    label=f'Uncorrected Reconstruction\nL2={err_before:.3e}'
)

ax1.plot(
    x_full,
    u_rec,
    color='green',
    lw=3,
    label=f'Reconstructed Solution\nL2={err_after:.3e}'
)

ax1.axvline(
    xI,
    color='black',
    linestyle=':',
    linewidth=2,
    label='Interface'
)

ax1.set_xlabel('Spatial Coordinate x')
ax1.set_ylabel('Solution u(x)')
ax1.grid(True)

# -------------------------------------------------
# Error axis
# -------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    x_full,
    u_unc - u_full,
    color='darkred',
    alpha=0.8,
    lw=2,
    label='Error Before Correction'
)

ax2.plot(
    x_full,
    u_rec - u_full,
    color='darkgreen',
    alpha=0.8,
    lw=2,
    label='Error After Correction'
)

ax2.set_ylabel('Pointwise Error')

# -------------------------------------------------
# Combined legend
# -------------------------------------------------

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='best'
)

ax1.set_title(
    'Solution Reconstruction and Pointwise Error\n'
    f'Interface Location = {xI:.2f}'
)

plt.tight_layout()

plt.savefig(
    run_dir / 'solution_with_error_flux.png',
    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)
# ============================================================
# EXPONENTIAL CORRECTION VS HIGH-ORDER COMPONENT
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(14, 10),
    sharex=False
)

# -------------------------------------------------
# Left side comparison
# -------------------------------------------------
ax1.plot(
    xL,
    uLh,
    lw=3,
    label=r'$u_{lH}$'
)

ax1.plot(
    xL,
    eL,
    'r--',
    lw=3,
    label=r'$e_L=a e^{\alpha x_L}$'
)

ax1.plot(
    xL,
    uLc,
    color='blue',
    lw=2,
    label=r'$uLc$'
)

ax1.set_title('Left Domain: Exponential Fit vs High-Order Contribution')
ax1.set_xlabel('x')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend()

# -------------------------------------------------
# Right side comparison
# -------------------------------------------------
ax2.plot(
    xR,
    uRh,
    lw=3,
    label=r'$u_{Rh}$'
)


ax2.plot(
    xR,
    eR,
    'r--',
    lw=3,
    label=r'$e_R=b e^{-\beta x_R}$'
)

ax2.plot(
    xR,
    uRc,
    color='blue',
    lw=2,
    label=r'$uRc$'
)

ax2.set_title('Right Domain: Exponential Fit vs High-Order Contribution')
ax2.set_xlabel('x')
ax2.set_ylabel('Amplitude')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

plt.savefig(
    run_dir / 'exponential_fit_comparison_flux.png',
    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)

Ju_before = Ju
Jq_before = Jflux

Ju_after = uLc[-1] - uRc[0]

Jq_after = (
    fluxLc
    -
    fluxRc
)

fig, ax = plt.subplots(
    figsize=(8,6)
)

labels = [
    "Solution jump",
    "Flux jump"
]

before = [
    abs(Ju_before),
    abs(Jq_before)
]

after = [
    abs(Ju_after),
    abs(Jq_after)
]

x = np.arange(len(labels))

ax.bar(
    x-0.2,
    before,
    width=0.4,
    label="Before"
)

ax.bar(
    x+0.2,
    after,
    width=0.4,
    label="After"
)

ax.set_yscale("log")

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.grid(True)

ax.legend()

plt.tight_layout()

plt.savefig(
    run_dir /
    "jump_reduction.png",
    dpi=600
)

residual_solution = (
    uLc[-1]
    -
    uRc[0]
)

residual_flux = (
    fluxLc
    -
    fluxRc
)

logger.info(
    "Interface residual solution = %.6e",
    residual_solution
)

logger.info(
    "Interface residual flux = %.6e",
    residual_flux
)

xI_values = np.linspace(
    5,
    95,
    500
)

cond_values = []

lam = np.sqrt(
    k_reaction / F0
)

for xI_test in xI_values:

    LW = xI_test
    LE = Lx - xI_test

    qW = (
        F0
        * lam
        / np.tanh(lam*LW)
    )

    qE = (
        -F0
        * lam
        / np.tanh(lam*LE)
    )

    Atest = np.array([
        [1,-1],
        [qW,-qE]
    ])

    cond_values.append(
        np.linalg.cond(Atest)
    )

plt.figure(figsize=(12,6))

plt.plot(
    xI_values,
    cond_values,
    lw=3
)

plt.yscale("log")

plt.xlabel("Interface location")

plt.ylabel("Condition number")

plt.grid(True)

plt.savefig(
    run_dir /
    "conditioning_analysis.png",
    dpi=600
)

logger.info("L2 error before correction = %.12e", err_before)
logger.info("L2 error after correction  = %.12e", err_after)

logger.debug(
    "Improvement factor = %.6f",
    err_before / err_after
    if err_after > 0 else np.inf
)

print('Results stored in',run_dir)
