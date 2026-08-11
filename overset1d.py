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

plt.rcParams.update({'font.size':14,'axes.titlesize':18,'axes.labelsize':16})

stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir=Path('runs')/f'variable_interface_{stamp}'
run_dir.mkdir(parents=True,exist_ok=True)

DEBUG_MODE = True

logger = logging.getLogger("validation")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

file_handler = logging.FileHandler(run_dir / "run.log")
file_handler.setFormatter(
    logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )
)

logger.addHandler(file_handler)

# Prevent messages from propagating to the root logger
logger.propagate = False

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
    return 1.5*np.sin(0.1*x)
    #return 0


def right_bc(x):
    return 1.5*np.sin(0.1*x)
    #return 0


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
ui_exact=np.interp(xI,x_full,u_full)
ul_guess=ui_exact + noise_amplitude*rng.standard_normal()
ur_guess=ui_exact + noise_amplitude*rng.standard_normal()


# SUBDOMAINS
xL,uL=solve_domain(0,xI,n_left,left_bc(0),ul_guess)
xR,uR=solve_domain(xI,Lx,n_right,ur_guess,right_bc(Lx))

# characteristic equation
x0Loc = -3
x1Loc = 2
x0 = xL[x0Loc]
x1 = xR[x1Loc]

Fp = F_prime(x0)
Fv = F_func(x0)
alpha1, alpha2 = np.roots([
    Fv,
    Fp,
    -k_reaction
])
Fp = F_prime(x1)
Fv = F_func(x1)
# Right characteristic roots
beta1, beta2 = np.roots([
    -Fv,
    Fp,
    k_reaction
])
alpha = alpha1 if np.real(alpha1) > np.real(alpha2) else alpha2
beta  = beta1  if np.real(beta1)  > np.real(beta2)  else beta2

logger.debug(f"alpha = {alpha:.6e}")
logger.debug(f"beta  = {beta:.6e}")


# exponentials referenced to the interface

a0 = np.exp(alpha * x0)
a1 = np.exp(alpha * x1)

b0 = np.exp(-beta *x0)
b1 = np.exp(-beta * x1)

uR1_old = u_full[n_left + x1Loc] 
uL0_old = u_full[n_left + x0Loc]

A = np.array([
    [1,0,0,b0],
    [0,1,a1,0],
    [0,0,a0,0],
    [0,0,0,b1]
])

rhs = np.array([
    uL0_old,
    uR1_old,
    uL[x0Loc] - uL0_old,
    uR[x1Loc] - uR1_old
])
u1, u2, a, b = np.linalg.solve(A, rhs)

# interface-centered homogeneous corrections


eL = a*np.exp(alpha*xL)
eR = b*np.exp(-beta*xR)

uLc = uL - eL
uRc = uR - eR

uLh = uL - u_full[:n_left+1]
uRh = uR - u_full[n_left:]

from scipy.optimize import curve_fit

def left_model(x, A, alpha):
    return A*np.exp(alpha*(x-xI))

popt, _ = curve_fit(
    left_model,
    xL,
    uLh,
    p0=(uLh[-1], 0.9)
)

A_fit, alpha_fit = popt

logger.debug(f"alpha_fit = {alpha_fit:.6e}")
logger.debug(f"a_fit = {A_fit:.6e}")
logger.debug(f"a = {a:.6e}")

def right_model(x, A, alpha):
    return A*np.exp(-alpha*(x-xI))

popt, _ = curve_fit(
    right_model,
    xL,
    uLh,
    p0=(uLh[-1], 0.9)
)

B_fit, beta_fit = popt

logger.debug(f"beta_fit = {beta_fit:.6e}")
logger.debug(f"b_fit = {B_fit:.6e}")
logger.debug(f"b = {b:.6e}")


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


logger.info("L2 error before correction = %.12e", err_before)
logger.info("L2 error after correction  = %.12e", err_after)

logger.debug(
    "Improvement factor = %.6f",
    err_before / err_after
    if err_after > 0 else np.inf
)

print('Results stored in',run_dir)
