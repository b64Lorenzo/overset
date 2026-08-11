#!/usr/bin/env python3
"""
2D hierarchical four-subdomain validation

           D3 | D4
          ---------
           D1 | D2

The four subdomains are solved independently using perturbed
interface data. Reconstruction is applied later through

    D1 <-> D2
    D3 <-> D4
    (D1+D2) <-> (D3+D4)

using the hierarchical reconstruction framework.
"""

from utils.logger import Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import (
    LinearNDInterpolator,
    interp1d,
)

from skfem import (
    BilinearForm,
    LinearForm,
)

from skfem.helpers import (
    dot,
    grad,
)

from utils.solvers import (
    solve_rect,
    compute_separation_modes,
)

from utils.interface_correction_paper import (
    InterfaceGeometry,
    InterfaceCorrector,
    CoupledInterfaceCorrector,
)

from utils.plot import (
    plot_field,
    plot_surface,
    plot_subdomains,
    export_subdomains,
    build_global_interpolator,
)

# ==========================================================
# PLOTTING
# ==========================================================

plt.rcParams.update(
    {
        "font.size": 14
    }
)

# ==========================================================
# LOGGER
# ==========================================================

log_mgr = Logger()

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()

# ==========================================================
# DOMAIN
# ==========================================================

Lx = 100.0
Ly = 40.0

xmid = Lx / 2.0
ymid = Ly / 2.0

# ==========================================================
# MESH RESOLUTION
# ==========================================================

nx_full = 200
ny_full = 300

nx_sub = nx_full // 2
ny_sub = ny_full // 2

# ==========================================================
# PDE PARAMETERS
# ==========================================================

k_reaction = 25.0

F0 = 30.0

P0 = 10.0
P_amp = 64.0

kxP = 0.64
kyP = 0.25

A_f = 1.0

fx_wave = 0.25
fy_wave = 0.15

F = 30.0

# ==========================================================
# COEFFICIENT FUNCTIONS
# ==========================================================

def P_func(x, y):

    return (
        P0
        + P_amp
        * np.cos(kxP * x)
        * np.cos(kyP * y)
    )


def Px_func(x, y):

    return (
        -P_amp
        * kxP
        * np.sin(kxP * x)
        * np.cos(kyP * y)
    )


def Py_func(x, y):

    return (
        -P_amp
        * kyP
        * np.cos(kxP * x)
        * np.sin(kyP * y)
    )


def fx(x, y):

    return 0.0


def fy(x, y):

    return 0.0


def fxx(x, y):

    return 0.0


def fyy(x, y):

    return 0.0


def forcing_term(x, y):

    return (
        Px_func(x, y) * fx(x, y)
        + Py_func(x, y) * fy(x, y)
        + P_func(x, y)
        * (fxx(x, y) + fyy(x, y))
    )

# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def bc_left(y):

    return 0.0


def bc_right(y):

    return 0.0


def bc_bottom(x):

    return 0.0


def bc_top(x):

    return 0.0

# ==========================================================
# FINITE ELEMENT FORMS
# ==========================================================

@BilinearForm
def diffusion(u, v, w):

    return F * dot(
        grad(u),
        grad(v)
    )


@BilinearForm
def reaction(u, v, w):

    return (
        k_reaction
        * u
        * v
    )


@LinearForm
def rhs(v, w):

    return (
        forcing_term(
            w.x[0],
            w.x[1]
        )
        * v
    )


# ==========================================================
# REFERENCE SOLUTION
# ==========================================================

logger.info(
    "Computing reference solution"
)

full_bc = {

    "left":
        lambda x, y: bc_left(y),

    "right":
        lambda x, y: bc_right(y),

    "bottom":
        lambda x, y: bc_bottom(x),

    "top":
        lambda x, y: bc_top(x),
}

meshF, basisF, uF = solve_rect(
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
        basisF.doflocs[1],
    ],
    uF,
)
prev_noise = 0.00064

def interp_prev_D1(x, y):
    return (
        interp_full(x, y)
        + prev_noise
        * np.sin(2.0*np.pi*x/Lx)
        * np.sin(np.pi*y/Ly)
    )

def interp_prev_D2(x, y):
    return (
        interp_full(x, y)
        - prev_noise
        * np.sin(2.0*np.pi*x/Lx)
        * np.sin(np.pi*y/Ly)
    )

def interp_prev_D3(x, y):
    return (
        interp_full(x, y)
        + prev_noise
        * np.cos(np.pi*x/Lx)
        * np.sin(2.0*np.pi*y/Ly)
    )

def interp_prev_D4(x, y):
    return (
        interp_full(x, y)
        - prev_noise
        * np.cos(np.pi*x/Lx)
        * np.sin(2.0*np.pi*y/Ly)
    )
# ==========================================================
# INTERFACE COORDINATES
# ==========================================================

y_bottom = np.linspace(
    0.0,
    ymid,
    ny_sub + 1,
)

y_top = np.linspace(
    ymid,
    Ly,
    ny_sub + 1,
)

x_interface = np.linspace(
    0.0,
    Lx,
    nx_full + 1,
)

# ==========================================================
# EXACT INTERFACE VALUES
# ==========================================================

ui12 = interp_full(
    np.full_like(y_bottom, xmid),
    y_bottom,
)

ui34 = interp_full(
    np.full_like(y_top, xmid),
    y_top,
)

uiH = interp_full(
    x_interface,
    np.full_like(
        x_interface,
        ymid
    ),
)

# ==========================================================
# NOISE
# ==========================================================

noise_amplitude = 10.0

u12_left = (
    ui12
    + noise_amplitude
    * np.sin(
        2.0 * np.pi * y_bottom / ymid
    )
)

u12_right = (
    ui12
    - noise_amplitude
    * np.sin(
        2.0 * np.pi * y_bottom / ymid
    )
)

u34_left = (
    ui34
    + noise_amplitude
    * np.sin(
        2.0*np.pi*(y_top-ymid)/ymid
    )
)

u34_right = (
    ui34
    - noise_amplitude
    * np.sin(
        2.0*np.pi*(y_top-ymid)/ymid
    )
)

u_lower = (
    uiH
    + noise_amplitude
    * np.sin(
        2.0*np.pi*x_interface/Lx
    )
)

u_upper = (
    uiH
    - noise_amplitude
    * np.sin(
        2.0*np.pi*x_interface/Lx
    )
)

# ==========================================================
# DOMAIN D1
# ==========================================================

logger.info("Solving D1")

bc1 = {

    "left":
        lambda x, y: bc_left(y),

    "bottom":
        lambda x, y: bc_bottom(x),

    "right":
    {
        "coords": y_bottom,
        "values": u12_left,
    },

    "top":
    {
        "coords": x_interface,
        "values": u_lower,
    },
}

mesh1, basis1, u1 = solve_rect(
    xmin=0.0,
    xmax=xmid,
    ymin=0.0,
    ymax=ymid,
    nx=nx_sub,
    ny=ny_sub,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=bc1,
)

# ==========================================================
# DOMAIN D2
# ==========================================================

logger.info("Solving D2")

bc2 = {

    "left":
    {
        "coords": y_bottom,
        "values": u12_right,
    },

    "right":
        lambda x, y: bc_right(y),

    "bottom":
        lambda x, y: bc_bottom(x),

    "top":
    {
        "coords": x_interface,
        "values": u_lower,
    },
}

mesh2, basis2, u2 = solve_rect(
    xmin=xmid,
    xmax=Lx,
    ymin=0.0,
    ymax=ymid,
    nx=nx_sub,
    ny=ny_sub,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=bc2,
)

# ==========================================================
# DOMAIN D3
# ==========================================================

logger.info("Solving D3")

bc3 = {

    "left":
        lambda x, y: bc_left(y),

    "bottom":
    {
        "coords": x_interface,
        "values": u_upper,
    },

    "right":
    {
        "coords": y_top,
        "values": u34_left,
    },

    "top":
        lambda x, y: bc_top(x),
}

mesh3, basis3, u3 = solve_rect(
    xmin=0.0,
    xmax=xmid,
    ymin=ymid,
    ymax=Ly,
    nx=nx_sub,
    ny=ny_sub,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=bc3,
)

# ==========================================================
# DOMAIN D4
# ==========================================================

logger.info("Solving D4")

bc4 = {

    "left":
    {
        "coords": y_top,
        "values": u34_right,
    },

    "right":
        lambda x, y: bc_right(y),

    "bottom":
    {
        "coords": x_interface,
        "values": u_upper,
    },

    "top":
        lambda x, y: bc_top(x),
}

mesh4, basis4, u4 = solve_rect(
    xmin=xmid,
    xmax=Lx,
    ymin=ymid,
    ymax=Ly,
    nx=nx_sub,
    ny=ny_sub,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=bc4,
)

# ==========================================================
# VERTICAL MODES
#
# D1 <-> D2
# D3 <-> D4
#
# Interface:
#     x = xmid
#
# Eigenfunctions:
#     sin(omega_vertical * y)
# ==========================================================

vertical_modes = compute_separation_modes(
    Ly=ymid,
    k_reaction=k_reaction,
    F=-F,
    n_modes=10,
)

vertical_indices = [0,1,2,3,4,5,6,7,8,9]

omegas_vertical = np.array([
    vertical_modes[i]["omega_y"]
    for i in vertical_indices
])

alphas_vertical = np.sqrt(
    omegas_vertical**2
    + k_reaction/F
)

# =====**=================================**================
# HORIZONTAL MOD**
#
# (D1+D2) <-> (D3+D4)
#
# Inte**ace:
#     y = ymid
#
# Eigenfunc**ons:
#     sin(omega_horizontal ***)
#
# Use Lx because the interfac**spans the
# entire domain width.
# ==========================================================

horizontal_modes = compute_separation_modes(
    Ly=Lx,
    k_reaction=k_reaction,
    F=-F,
    n_modes=10,
)

horizontal_indices = [0,1,2,3,4,5,6,7,8,9]

omegas_horizontal = np.array([
    horizontal_modes[i]["omega_y"]
    for i in horizontal_indices
])

alphas_horizontal = np.sqrt(
    omegas_horizontal**2
    + k_reaction/F
)
# ==========================================================
# LOGGING
# ==========================================================

logger.info("")
logger.info("================================")
logger.info("VERTICAL INTERFACE MODE")
logger.info("================================")

logger.info(
    f"vertical omegas = "
    f"{omegas_vertical}"
)

logger.info(
    f"vertical alphas = "
    f"{alphas_vertical}"
)

logger.info("")
logger.info("================================")
logger.info("HORIZONTAL INTERFACE MODE")
logger.info("================================")
logger.info(
    f"horizontal omegas = "
    f"{omegas_horizontal}"
)

logger.info(
    f"horizontal alphas = "
    f"{alphas_horizontal}"
)


# ==========================================================
# GENERIC INTERFACE CORRECTION
# ==========================================================



geom12 = InterfaceGeometry(

    interface_type="vertical",

    location=xmid,

    decay_rates=alphas_vertical,

    omegas=omegas_vertical,
)

res12 = InterfaceCorrector.correct(

    basisA=basis1,
    uA=u1,

    basisB=basis2,
    uB=u2,

    interp_previous_A=interp_prev_D1,
    interp_previous_B=interp_prev_D2,

    geometry=geom12,
)

geom34 = InterfaceGeometry(

    interface_type="vertical",

    location=xmid,

    decay_rates=alphas_vertical,

    omegas=omegas_vertical,
)

res34 = InterfaceCorrector.correct(

    basisA=basis3,
    uA=u3,

    basisB=basis4,
    uB=u4,

    interp_previous_A=interp_prev_D3,
    interp_previous_B=interp_prev_D4,

    geometry=geom34,
)

u1v = res12.corrected_A
u2v = res12.corrected_B

u3v = res34.corrected_A
u4v = res34.corrected_B



subdomains_stage1 = [

    {
        "name": "D1",
        "basis": basis1,
        "uc": u1v,
    },

    {
        "name": "D2",
        "basis": basis2,
        "uc": u2v,
    },

    {
        "name": "D3",
        "basis": basis3,
        "uc": u3v,
    },

    {
        "name": "D4",
        "basis": basis4,
        "uc": u4v,
    },
]

interp_stage1 = build_global_interpolator(
    subdomains_stage1,
    "uc",
)

Xf = basisF.doflocs[0]
Yf = basisF.doflocs[1]

u_stage1 = interp_stage1(
    Xf,
    Yf,
)

err_stage1 = u_stage1 - uF


plot_field(
    Xf,
    Yf,
    err_stage1,
    "Error After Stage 1",
    "error_after_stage1.png",
    run_dir,
    cmap="seismic",
)

L2_stage1 = np.linalg.norm(
    err_stage1
)

logger.info(
    f"L2 after stage 1 = "
    f"{L2_stage1:.6e}"
)

# ==========================================================
# COUPLED RECONSTRUCTION
# ==========================================================

res1 = CoupledInterfaceCorrector.fit(

    basis=basis1,

    u=u1,

    interp_previous=interp_prev_D1,

    xI=xmid,

    yI=ymid,

    alphas_x=alphas_vertical,
    omegas_x=omegas_vertical,

    alphas_y=alphas_horizontal,
    omegas_y=omegas_horizontal,

)

res2 = CoupledInterfaceCorrector.fit(

    basis=basis2,

    u=u2,

    interp_previous=interp_prev_D2,

    xI=xmid,

    yI=ymid,

    alphas_x=alphas_vertical,
    omegas_x=omegas_vertical,

    alphas_y=alphas_horizontal,
    omegas_y=omegas_horizontal,

)

res3 = CoupledInterfaceCorrector.fit(

    basis=basis3,

    u=u3,

    interp_previous=interp_prev_D3,

    xI=xmid,

    yI=ymid,

    alphas_x=alphas_vertical,
    omegas_x=omegas_vertical,

    alphas_y=alphas_horizontal,
    omegas_y=omegas_horizontal,

)

res4 = CoupledInterfaceCorrector.fit(

    basis=basis4,

    u=u4,

    interp_previous=interp_prev_D4,

    xI=xmid,

    yI=ymid,

    alphas_x=alphas_vertical,
    omegas_x=omegas_vertical,

    alphas_y=alphas_horizontal,
    omegas_y=omegas_horizontal,

)



u1f = res1.corrected

u2f = res2.corrected

u3f = res3.corrected

u4f = res4.corrected

corr1 = (
    res1.correction
)

corr2 = (
    
    res2.correction
)

corr3 = (
    
    res3.correction
)

corr4 = (
    
    res4.correction
)

subdomains = [

    {
        "name": "D1",
        "basis": basis1,
        "u": u1,
        "uc": u1f,
        "correction": corr1,
    },

    {
        "name": "D2",
        "basis": basis2,
        "u": u2,
        "uc": u2f,
        "correction": corr2,
    },

    {
        "name": "D3",
        "basis": basis3,
        "u": u3,
        "uc": u3f,
        "correction": corr3,
    },

    {
        "name": "D4",
        "basis": basis4,
        "u": u4,
        "uc": u4f,
        "correction": corr4,
    },
]
logger.info(
    f"A12={res12.coefficient_A}"
)

logger.info(
    f"B12={res12.coefficient_B}"
)

logger.info(
    f"||traceA12||="
    f"{np.linalg.norm(res12.trace_A):.6e}"
)

logger.info(
    f"||traceB12||="
    f"{np.linalg.norm(res12.trace_B):.6e}"
)
# ==========================================================
# GLOBAL RECONSTRUCTION
# ==========================================================

interp_unc = build_global_interpolator(
    subdomains,
    "u",
)

interp_rec = build_global_interpolator(
    subdomains,
    "uc",
)

Xf = basisF.doflocs[0]
Yf = basisF.doflocs[1]

u_unc = interp_unc(
    Xf,
    Yf,
)

u_rec = interp_rec(
    Xf,
    Yf,
)

err_before = u_unc - uF
err_after = u_rec - uF

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
    f"Improvement = "
    f"{L2_before/L2_after:.3f}"
)

# ==========================================================
# SUBDOMAIN PLOTS
# ==========================================================

plot_subdomains(
    subdomains,
    run_dir,
)

# ==========================================================
# GLOBAL PLOTS
# ==========================================================

plot_surface(
    Xf,
    Yf,
    uF,
    "Reference Solution",
    "reference_3d.png",
    run_dir,
)

plot_field(
    Xf,
    Yf,
    uF,
    "Reference Solution",
    "reference_2d.png",
    run_dir,
)

plot_surface(
    Xf,
    Yf,
    u_rec,
    "Reconstructed Solution",
    "reconstructed_3d.png",
    run_dir,
)

plot_field(
    Xf,
    Yf,
    u_rec,
    "Reconstructed Solution",
    "reconstructed_2d.png",
    run_dir,
)

plot_field(
    Xf,
    Yf,
    err_before,
    "Error Before",
    "error_before.png",
    run_dir,
    cmap="seismic",
)

plot_field(
    Xf,
    Yf,
    err_after,
    "Error After",
    "error_after.png",
    run_dir,
    cmap="seismic",
)

# ==========================================================
# EXPORTS
# ==========================================================

export_subdomains(
    subdomains,
    run_dir,
)

pd.DataFrame(
    {
        "x": Xf,
        "y": Yf,
        "u_reference": uF,
        "u_uncorrected": u_unc,
        "u_reconstructed": u_rec,
        "error_before": err_before,
        "error_after": err_after,
    }
).to_csv(
    run_dir / "global_validation.csv",
    index=False,
)

print(
    f"Results stored in {run_dir}"
)