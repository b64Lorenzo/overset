#!/usr/bin/env python3
"""
flux1d.py
=========

1D Domain-Decomposition Validation

Solves

    -(h(x)u_x)_x + k u = d/dx(p f_x)

using finite elements.

The domain is decomposed into two subdomains
separated by an interface.

A correction procedure reconstructs:

    - solution continuity
    - flux continuity

using homogeneous basis functions.

Author
------
Lorenzo Zambelli
"""

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import Logger

from src.solver.fem1d import FEMEllipticSolver1D

from src.problems.elliptic1d import (
    Elliptic1DProblem,
)

from src.utils.utils import (
    derivative_west,
    derivative_east,
    interface_flux,
    reconstruct_solution,
)

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    }
)


# ============================================================
# LOGGER
# ============================================================

log_mgr = Logger(
    name="flux1d",
    run_prefix="flux1d",
)

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()


# ============================================================
# CONFIGURATION
# ============================================================

Lx = 100.0

xI = 64.0

n_full = 800

n_west = 400
n_east = 400

k_reaction = 25.0

noise_amplitude = 6.4

current_time = 0.0

rng = np.random.default_rng(42)


# ============================================================
# PDE COEFFICIENTS
# ============================================================

F0 = 30.0

P0 = 10.0
P_amp = 64.0
k_p = 0.64

A_f = 1.0
k_f = 0.25

omega = 0.64


def F_func(x):
    return (
        F0
        + 5.0 * np.cos(0.64 * x)
    )


def F_prime(x):

    return (
        -5.0
        * 0.64
        * np.sin(0.64 * x)
    )


def P_func(x):

    return (
        P0
        + P_amp * np.cos(k_p * x)
    )


def P_prime(x):

    return (
        -P_amp
        * k_p
        * np.sin(k_p * x)
    )


def f_x(x, t):

    return (
        A_f
        * k_f
        * np.cos(
            k_f * x
            - omega * t
        )
    )


def f_xx(x, t):

    return (
        -A_f
        * k_f**2
        * np.sin(
            k_f * x
            - omega * t
        )
    )


def left_bc(x):
    return 0.0


def right_bc(x):
    return 0.0


problem = Elliptic1DProblem(
    k=k_reaction,
    h=F_func,
    h_x=F_prime,
    p=P_func,
    p_x=P_prime,
    f_x=f_x,
    f_xx=f_xx,
)

solver = FEMEllipticSolver1D(problem)


# ============================================================
# BENCHMARK SOLUTION
# ============================================================

x_full, u_full = solver.solve(
    a=0.0,
    b=Lx,
    n_elements=n_full,
    bc_left=0.0,
    bc_right=0.0,
    current_time=current_time,
)

ui_exact = np.interp(
    xI,
    x_full,
    u_full,
)

ul_guess = (
    ui_exact
    + noise_amplitude
    * rng.standard_normal()
)

ur_guess = (
    ui_exact
    + noise_amplitude
    * rng.standard_normal()
)

logger.info(
    "Exact interface value = %.6e",
    ui_exact,
)

logger.info(
    "West interface guess = %.6e",
    ul_guess,
)

logger.info(
    "East interface guess = %.6e",
    ur_guess,
)


# ============================================================
# SUBDOMAIN SOLVES
# ============================================================

xL, u_west = solver.solve(
    a=0.0,
    b=xI,
    n_elements=n_west,
    bc_left=0.0,
    bc_right=ul_guess,
    current_time=current_time,
)

xR, u_east = solver.solve(
    a=xI,
    b=Lx,
    n_elements=n_east,
    bc_left=ur_guess,
    bc_right=0.0,
    current_time=current_time,
)
# ============================================================
# INTERFACE JUMPS
# ============================================================

dx_west = (
    xL[-1]
    - xL[-2]
)

dx_east = (
    xR[1]
    - xR[0]
)

Ju = (
    u_west[-1]
    - u_east[0]
)

flux_west, flux_east = interface_flux(
    u_west=u_west,
    u_east=u_east,
    dx_west=dx_west,
    dx_east=dx_east,
    x_interface=xI,
    diffusion_coefficient=F_func,
)

Jflux = (
    flux_west
    - flux_east
)

logger.info("Initial solution jump = %.6e",Ju)

logger.info("Initial flux jump = %.6e",Jflux)


# ============================================================
# HOMOGENEOUS MODES
# ============================================================

_, phi_west = solver.solve(
    a=0.0,
    b=xI,
    n_elements=n_west,
    bc_left=0.0,
    bc_right=1.0,
    homogeneous=True,
)

_, phi_east = solver.solve(
    a=xI,
    b=Lx,
    n_elements=n_east,
    bc_left=1.0,
    bc_right=0.0,
    homogeneous=True,
)


qphi_west, qphi_east = interface_flux(
    u_west=phi_west,
    u_east=phi_east,
    dx_west=dx_west,
    dx_east=dx_east,
    x_interface=xI,
    diffusion_coefficient=F_func,
)


A = np.array(
    [
        [1.0, -1.0],
        [qphi_west, -qphi_east],
    ]
)

rhs = np.array(
    [
        Ju,
        Jflux,
    ]
)

eps_west, eps_east = np.linalg.solve(
    A,
    rhs,
)

condA = np.linalg.cond(A)

logger.info(
    "Cond(A)=%.6e",
    condA,
)


e_west = eps_west * phi_west
e_east = eps_east * phi_east

u_west_corrected = u_west - e_west
u_east_corrected = u_east - e_east
# ============================================================
# RECONSTRUCTION
# ============================================================

u_unc = reconstruct_solution(
    x_full,
    xI,
    xL,
    u_west,
    xR,
    u_east,
)

u_rec = reconstruct_solution(
    x_full,
    xI,
    xL,
    u_west_corrected,
    xR,
    u_east_corrected,
)

err_before = np.linalg.norm(
    u_unc - u_full
)

err_after = np.linalg.norm(
    u_rec - u_full
)

logger.info(
    "L2 error before correction = %.12e",
    err_before,
)

logger.info(
    "L2 error after correction = %.12e",
    err_after,
)


flux_west_corrected, flux_east_corrected = interface_flux(
    u_west=u_west_corrected,
    u_east=u_east_corrected,
    dx_west=dx_west,
    dx_east=dx_east,
    x_interface=xI,
    diffusion_coefficient=F_func,
)

Ju_after = (
    u_west_corrected[-1]
    - u_east_corrected[0]
)

Jflux_after = (
    flux_west_corrected
    - flux_east_corrected
)

logger.info(
    "Interface solution residual = %.6e",
    Ju_after,
)

logger.info(
    "Interface flux residual = %.6e",
    Jflux_after,
)

lhsf, rhsf, resf = solver.residual(
    x_full,
    u_full,
    current_time=current_time,
)

# ============================================================
# HOMOGENEOUS CORRECTION VALIDATION
# ============================================================

u_exact_left = np.interp(
    xL,
    x_full,
    u_full,
)

u_exact_right = np.interp(
    xR,
    x_full,
    u_full,
)

true_homogeneous_left = (
    u_west
    - u_exact_left
)

true_homogeneous_right = (
    u_east
    - u_exact_right
)

left_h_error = np.linalg.norm(
    true_homogeneous_left
    - e_west
)

right_h_error = np.linalg.norm(
    true_homogeneous_right
    - e_east
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(13, 10),
)

ax1.plot(
    xL,
    true_homogeneous_left,
    lw=3,
    label="True correction",
)

ax1.plot(
    xL,
    e_west,
    "--",
    lw=3,
    label="Reconstructed correction",
)

ax1.text(
    0.02,
    0.95,
    f"L2={left_h_error:.3e}",
    transform=ax1.transAxes,
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
    label="True correction",
)

ax2.plot(
    xR,
    e_east,
    "--",
    lw=3,
    label="Reconstructed correction",
)

ax2.text(
    0.02,
    0.95,
    f"L2={right_h_error:.3e}",
    transform=ax2.transAxes,
)

ax2.grid(True)
ax2.legend()

ax2.set_title(
    "East Homogeneous Correction"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "homogeneous_correction.png",
    dpi=600,
)

plt.close()

# ============================================================
# SOLUTION RECONSTRUCTION
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(16, 9)
)

ax1.plot(
    x_full,
    u_full,
    color="black",
    lw=4,
    label="Benchmark",
)

ax1.plot(
    x_full,
    u_unc,
    "--",
    color="red",
    lw=2,
    label=f"Uncorrected (L2={err_before:.3e})",
)

ax1.plot(
    x_full,
    u_rec,
    color="green",
    lw=3,
    label=f"Corrected (L2={err_after:.3e})",
)

ax1.axvline(
    xI,
    color="black",
    linestyle=":",
    linewidth=2,
    label="Interface",
)

ax1.set_xlabel("x")
ax1.set_ylabel("Solution")

ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    x_full,
    u_unc - u_full,
    color="darkred",
    lw=2,
    alpha=0.8,
    label="Error before",
)

ax2.plot(
    x_full,
    u_rec - u_full,
    color="darkgreen",
    lw=2,
    alpha=0.8,
    label="Error after",
)

ax2.set_ylabel("Pointwise Error")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
)

ax1.set_title(
    "Solution Reconstruction"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "solution_reconstruction.png",
    dpi=600,
)

plt.close()


# ============================================================
# HIGH-ORDER COMPONENT COMPARISON
# ============================================================

uLh = (
    u_west
    - u_exact_left
)

uRh = (
    u_east
    - u_exact_right
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(14, 10),
)

ax1.plot(
    xL,
    uLh,
    lw=3,
    label=r"$u_{LH}$",
)

ax1.plot(
    xL,
    e_west,
    "r--",
    lw=3,
    label=r"$e_L$",
)

ax1.plot(
    xL,
    u_west_corrected,
    lw=2,
    color="blue",
    label=r"$u_{Lc}$",
)

ax1.grid(True)
ax1.legend()

ax1.set_title(
    "West Domain"
)

ax2.plot(
    xR,
    uRh,
    lw=3,
    label=r"$u_{RH}$",
)

ax2.plot(
    xR,
    e_east,
    "r--",
    lw=3,
    label=r"$e_R$",
)

ax2.plot(
    xR,
    u_east_corrected,
    lw=2,
    color="blue",
    label=r"$u_{Rc}$",
)

ax2.grid(True)
ax2.legend()

ax2.set_title(
    "East Domain"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "high_order_comparison.png",
    dpi=600,
)

plt.close()

# ============================================================
# JUMP REDUCTION
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

labels = [
    "Solution jump",
    "Flux jump",
]

before = [
    abs(Ju),
    abs(Jflux),
]

after = [
    abs(Ju_after),
    abs(Jflux_after),
]

xbar = np.arange(
    len(labels)
)

ax.bar(
    xbar - 0.2,
    before,
    width=0.4,
    label="Before",
)

ax.bar(
    xbar + 0.2,
    after,
    width=0.4,
    label="After",
)

ax.set_yscale("log")

ax.set_xticks(xbar)

ax.set_xticklabels(
    labels
)

ax.grid(True)

ax.legend()

ax.set_title(
    "Interface Jump Reduction"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "jump_reduction.png",
    dpi=600,
)

plt.close()

# ============================================================
# CONDITIONING STUDY
# ============================================================

xI_values = np.linspace(
    5.0,
    95.0,
    500,
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
        / np.tanh(lam * LW)
    )

    qE = (
        -F0
        * lam
        / np.tanh(lam * LE)
    )

    Atest = np.array(
        [
            [1.0, -1.0],
            [qW, -qE],
        ]
    )

    cond_values.append(
        np.linalg.cond(
            Atest
        )
    )

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    xI_values,
    cond_values,
    lw=3,
)

ax.set_yscale("log")

ax.grid(True)

ax.set_xlabel(
    "Interface location"
)

ax.set_ylabel(
    "Condition number"
)

ax.set_title(
    "Interface Conditioning"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "conditioning_analysis.png",
    dpi=600,
)

plt.close()

# ============================================================
# RESIDUAL PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    x_full,
    resf,
    lw=2,
)

ax.grid(True)

ax.set_xlabel("x")

ax.set_ylabel("Residual")

ax.set_title(
    "PDE Residual"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "pde_residual.png",
    dpi=600,
)

plt.close()

print()
print("=" * 60)
print("FLUX1D COMPLETED")
print("=" * 60)
print(f"L2 before : {err_before:.6e}")
print(f"L2 after  : {err_after:.6e}")
print(f"Cond(A)   : {condA:.6e}")
print(f"Results   : {run_dir}")
print("=" * 60)