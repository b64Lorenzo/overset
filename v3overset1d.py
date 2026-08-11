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
from utils.interface_correction_1d import *
from utils.logger import Logger

plt.rcParams.update({'font.size':14,'axes.titlesize':18,'axes.labelsize':16})


log_mgr = Logger(
    name="v3overset1d",
    run_prefix="v31d_fx",
)

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
h_left  = xI / n_left
h_right = (Lx - xI) / n_right
delta_left = 3
delta_right = 1 
xI_left = xI + delta_left * h_left
xI_right = xI - delta_right * h_right



k_reaction = 25.0
relax = 0.64
noise_amplitude = 30
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
    
    logger.debug(
            "solve_domain(a=%s, b=%s, n=%s, bc_left=%s, bc_right=%s, homogeneous=%s)",
            a, b, n, bc_left, bc_right, homogeneous
        )

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
    
    
    logger.debug(
            "solve_domain finished: u.min=%e u.max=%e",
            np.min(u),
            np.max(u)
        )

    return mesh.p[0],u


def pde_balance(x,u):
    dx=np.mean(np.diff(x))
    ux=np.gradient(u,dx,edge_order=2)
    flux=F_func(x)*ux
    fluxx=np.gradient(flux,dx,edge_order=2)
    lhs=-fluxx+k_reaction*u
    rhsv=forcing_term(x)
    return lhs,rhsv,lhs-rhsv
    
def reconstruct_overset_1d(
    x_eval,
    xL,
    uL,
    xR,
    uR,
    xI,
    xI_left=None,
    xI_right=None,
    assembly="sharp",
):
    """
    Assemble a global 1D solution from two overset subdomains.

    Parameters
    ----------
    x_eval : ndarray
        Coordinates where the global solution is evaluated.

    xL, uL : ndarray
        Left subdomain solution.

    xR, uR : ndarray
        Right subdomain solution.

    xI : float
        Physical interface location.

    xI_left : float
        Right boundary of left overset domain.

    xI_right : float
        Left boundary of right overset domain.

    assembly : str

        "sharp"
            Cut at xI.

        "blend"
            Blend in overlap region.

    Returns
    -------
    ndarray
        Global reconstructed solution.
    """

    x_eval = np.asarray(x_eval)

    interpL = lambda x: np.interp(x, xL, uL)
    interpR = lambda x: np.interp(x, xR, uR)

    u = np.zeros_like(
        x_eval,
        dtype=float,
    )

    # --------------------------------------------------
    # SHARP
    # --------------------------------------------------

    if assembly == "sharp":

        maskL = x_eval <= xI
        maskR = x_eval > xI

        if np.any(maskL):
            u[maskL] = interpL(
                x_eval[maskL]
            )

        if np.any(maskR):
            u[maskR] = interpR(
                x_eval[maskR]
            )

        return u

    # --------------------------------------------------
    # BLEND
    # --------------------------------------------------

    if assembly == "blend":

        if (
            xI_left is None
            or
            xI_right is None
        ):
            raise ValueError(
                "blend assembly requires "
                "xI_left and xI_right"
            )

        maskL = (
            x_eval < xI_right
        )

        maskR = (
            x_eval > xI_left
        )

        maskO = (
            (x_eval >= xI_right)
            &
            (x_eval <= xI_left)
        )

        if np.any(maskL):

            u[maskL] = interpL(
                x_eval[maskL]
            )

        if np.any(maskR):

            u[maskR] = interpR(
                x_eval[maskR]
            )

        if np.any(maskO):

            wL = (
                xI_left
                - x_eval[maskO]
            ) / (
                xI_left
                - xI_right
            )

            wR = 1.0 - wL

            u[maskO] = (
                wL
                * interpL(
                    x_eval[maskO]
                )
                +
                wR
                * interpR(
                    x_eval[maskO]
                )
            )

        return u

    raise ValueError(
        f"Unknown assembly '{assembly}'"
    )

if logger.isEnabledFor(logging.DEBUG):
    logger.debug("========== RUN PARAMETERS ==========")

    params = {
        "Lx": Lx,
        "xI": xI,
        "n_full": n_full,
        "n_left": n_left,
        "n_right": n_right,
        "k_reaction": k_reaction,
        "relax": relax,
        "noise_amplitude": noise_amplitude,
        "F0": F0,
        "P0": P0,
        "P_amp": P_amp,
        "k_p": k_p,
        "A_f": A_f,
        "k_f": k_f,
        "omega": omega,
        "current_time": current_time,
    }

    for k, v in params.items():
        logger.debug("%s = %s", k, v)

    logger.debug("====================================")

# BENCHMARK
x_full,u_full=solve_domain(0,Lx,n_full,left_bc(0),right_bc(Lx))
ui_exact_left=np.interp(xI_left,x_full,u_full)
ui_exact_right=np.interp(xI_right,x_full,u_full)
ul_guess=noise_amplitude*ui_exact_left 
ur_guess=noise_amplitude*ui_exact_right


# SUBDOMAINS
xL,uL=solve_domain(0,xI_left,n_left,left_bc(0),ul_guess)
xR,uR=solve_domain(xI_right,Lx,n_right,ur_guess,right_bc(Lx))

# decay ration 
F_overlap = np.mean(
    F_func(np.linspace(xI_right, xI_left, 100))
)
lam = np.sqrt(k_reaction / F_overlap)
lam = np.sqrt(k_reaction / F_func(xI))


phiL = lambda x: np.exp(lam * (x - xI_left))
phiR = lambda x: np.exp(-lam * (x - xI_right))

logger.info(f"Approx constant F-> lamda={lam}")
# exponentials referenced to the interface

aL = phiL(xI_left)     # = 1
aR = phiL(xI_right)

bL = phiR(xI_left)
bR = phiR(xI_right)    # = 1

uRl = np.interp(xI_left,xR,uR)
uLr = np.interp(xI_right,xL,uL)

A = np.array([
    [1,0,0,bL],
    [0,1,aR,0],
    [1,0,aL,0],
    [0,1,0,bR]
])

rhs = np.array([
    uRl,
    uLr,
    ul_guess,
    ur_guess
])
u1, u2, a, b = np.linalg.solve(A, rhs)

# interface-centered homogeneous corrections

eL = a * phiL(xL)
eR = b * phiR(xR)



# ============================================================
# CORRECTED SOLUTIONS
# ============================================================

uLc = uL - eL
uRc = uR - eR

logger.debug(
    "reconstruction complete"
)

logger.debug(
    "||eL|| = %.6e",
    np.linalg.norm(eL)
)

logger.debug(
    "||eR|| = %.6e",
    np.linalg.norm(eR)
)

uLh = uL - u_full[:n_left+1]
uRh = uR - u_full[n_left:]



#assembly = "blend"
assembly = "sharp"

u_unc = reconstruct_overset_1d(
    x_full,
    xL,
    uL,
    xR,
    uR,
    xI=xI,
    xI_left=xI_left,
    xI_right=xI_right,
    assembly=assembly,
)

u_rec = reconstruct_overset_1d(
    x_full,
    xL,
    uLc,
    xR,
    uRc,
    xI=xI,
    xI_left=xI_left,
    xI_right=xI_right,
    assembly=assembly,
)

err_before = np.linalg.norm(
    u_unc - u_full
)

err_after = np.linalg.norm(
    u_rec - u_full
)

logger.info(
    f"Assembly = {assembly}"
)

logger.info(
    f"L2 error before = {err_before:.6e}"
)

logger.info(
    f"L2 error after  = {err_after:.6e}"
)

logger.info(
    f"Improvement = "
    f"{err_before/err_after:.3f}"
)

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
    run_dir / 'solution_with_error.png',
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
    run_dir / 'exponential_fit_comparison.png',
    dpi=600,
    bbox_inches='tight'
)

plt.close(fig)

pd.DataFrame({'x':x_full,'u_full':u_full,'u_reconstructed':u_rec}).to_csv(run_dir/'results.csv',index=False)



logger.debug(
    "Improvement factor = %.6f",
    err_before / err_after
    if err_after > 0 else np.inf
)

print('Results stored in',run_dir)

# ============================================================
# OVERLAP ZOOM - SHOW BOTH SUBDOMAINS
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

pad = 2.0

xmin = xI_right - pad
xmax = xI_left + pad

maskL = (xL >= xmin) & (xL <= xmax)
maskR = (xR >= xmin) & (xR <= xmax)
maskF = (x_full >= xmin) & (x_full <= xmax)

# exact
ax.plot(
    x_full[maskF],
    u_full[maskF],
    'k',
    lw=4,
    label='Exact'
)

# left domain
ax.plot(
    xL[maskL],
    uL[maskL],
    'r--',
    lw=3,
    label='Left uncorrected'
)

ax.plot(
    xL[maskL],
    uLc[maskL],
    'r',
    lw=3,
    label='Left corrected'
)

# right domain
ax.plot(
    xR[maskR],
    uR[maskR],
    'b--',
    lw=3,
    label='Right uncorrected'
)

ax.plot(
    xR[maskR],
    uRc[maskR],
    'b',
    lw=3,
    label='Right corrected'
)

# overlap visualization
ax.axvspan(
    xI_right,
    xI_left,
    color='gold',
    alpha=0.25,
    label='Overlap'
)

ax.axvline(
    xI_right,
    color='grey',
    linestyle='--'
)

ax.axvline(
    xI_left,
    color='grey',
    linestyle='--'
)

ax.set_xlim(xmin, xmax)

ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.set_title("Overlap Region Detail")
ax.grid(True)
ax.legend()

plt.tight_layout()

plt.savefig(
    run_dir / "overlap_detail.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()
