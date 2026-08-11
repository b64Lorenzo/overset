#!/usr/bin/env python3


from utils.logger import Logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.solvers import solve_rect, compute_separation_modes
from scipy.interpolate import LinearNDInterpolator, interp1d
from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad
from utils.overset_interface_corrector_rewritten import (
    compute_modes,
    OversetInterfaceCorrector2D,
)
from utils.plot import (
    build_global_interpolator,
)

plt.rcParams.update({'font.size':14})

log_mgr = Logger()

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()

nx_full = 200
ny_full = 300

nx_left = 300
nx_right = 300

Lx = 100.0
Ly = 40.0

xI = 64.0

overlap_left = 10
overlap_right = 10

hxL = xI / nx_left
hxR = (Lx - xI) / nx_right

xI_left = xI + overlap_left * hxL
xI_right = xI - overlap_right * hxR

k_reaction=25.0

F0=30.0
P0=10.0
P_amp=64.0
kxP=0.64
kyP=0.25

A_f=1.0
fx_wave=0.25
fy_wave=0.15
F0 = 30.0

def F_func(x, y):
    return (
        F0
        + 5.0*np.cos(0.64*x)
        + 2.0*np.sin(0.25*y)
    )

def P_func(x,y):
    return P0+P_amp*np.cos(kxP*x)*np.cos(kyP*y)

def Px_func(x,y):
    return -P_amp*kxP*np.sin(kxP*x)*np.cos(kyP*y)

def Py_func(x,y):
    return -P_amp*kyP*np.cos(kxP*x)*np.sin(kyP*y)

def fx(x,y):
    return A_f*fx_wave*np.cos(fx_wave*x)*np.sin(fy_wave*y)
    #return 0

def fy(x,y):
    return A_f*fy_wave*np.sin(fx_wave*x)*np.cos(fy_wave*y)
    #return 0

def fxx(x,y):
    return -A_f*(fx_wave**2)*np.sin(fx_wave*x)*np.sin(fy_wave*y)
    #return 0

def fyy(x,y):
    return -A_f*(fy_wave**2)*np.sin(fx_wave*x)*np.sin(fy_wave*y)
    #return 0

def forcing_term(x,y):
    return Px_func(x,y)*fx(x,y)+Py_func(x,y)*fy(x,y)+P_func(x,y)*(fxx(x,y)+fyy(x,y))

def bc(x,y):
    return 1.5*np.sin(0.1*x)*np.cos(0.1*y)
    
def bc_left(y):
    #return 1.5*np.cos(0.1*y)
    return 0

def bc_right(y):
    #return 1.5*np.cos(0.1*y)
    return 0

def bc_bottom(x):
    #return 1.5*np.sin(0.1*x)
    return 0

def bc_top(x):
    #return 1.5*np.sin(0.1*x)  
    return 0

@BilinearForm
def diffusion(u,v,w):

    return (
        F_func(
            w.x[0],
            w.x[1],
        )
        *
        dot(
            grad(u),
            grad(v),
        )
    )

@BilinearForm
def reaction(u,v,w):
    return k_reaction*u*v

@LinearForm
def rhs(v,w):
    return forcing_term(w.x[0],w.x[1])*v



full_bc = {
    "left": lambda x, y: bc_left(y),
    "right": lambda x, y: bc_right(y),
    "bottom": lambda x, y: bc_bottom(x),
    "top": lambda x, y: bc_top(x),
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
        basisF.doflocs[1]
    ],
    uF
)
y_interface = np.linspace(
    0,
    Ly,
    ny_full + 1
)
ui_exact_left = interp_full(
    np.full_like(
        y_interface,
        xI_left,
    ),
    y_interface,
)

ui_exact_right = interp_full(
    np.full_like(
        y_interface,
        xI_right,
    ),
    y_interface,
)
rng = np.random.default_rng(42)


noise_amplitude = 10

ul_guess = (
    ui_exact_left
    +
    noise_amplitude
    *
    np.sin(
        2*np.pi*y_interface/Ly
    )
)

ur_guess = (
    ui_exact_right
    +
    noise_amplitude
    *
    np.sin(
        -2*np.pi*y_interface/Ly
    )
)

logger.info("solving for each subdomain")

left_bc = {
    "left": lambda x, y: bc_left(y),

    "bottom": lambda x, y: bc_bottom(x),

    "top": lambda x, y: bc_top(x),

    "right": {
        "coords": y_interface,
        "values": ul_guess,
    },
}

meshL, basisL, uL = solve_rect(
    xmin=0.0,
    xmax=xI_left,
    ymin=0.0,
    ymax=Ly,
    nx=nx_left,
    ny=ny_full,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=left_bc,
)

right_bc = {
    "left": {
        "coords": y_interface,
        "values": ur_guess,
    },

    "right": lambda x, y: bc_right(y),

    "bottom": lambda x, y: bc_bottom(x),

    "top": lambda x, y: bc_top(x),
}

meshR, basisR, uR = solve_rect(
    xmin=xI_right,
    xmax=Lx,
    ymin=0.0,
    ymax=Ly,
    nx=nx_right,
    ny=ny_full,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions=right_bc,
)

logger.info("Computing modes")

# ==========================================================
# OVERSET MODES
# ==========================================================

ys = np.linspace(
    0.0,
    Ly,
    400,
)

F_interface = np.mean(
    F_func(
        np.full_like(
            ys,
            xI,
        ),
        ys,
    )
)

alphas, omegas = compute_modes(
    Ly=Ly,
    k_reaction=k_reaction,
    F_interface=F_interface,
    n_modes=10,
)

logger.info(
    f"F_interface = {F_interface:.6e}"
)

for n, (a, w) in enumerate(
    zip(alphas, omegas),
    start=1,
):
    logger.info(
        f"mode={n:2d} "
        f"omega={w:.6e} "
        f"alpha={a:.6e}"
    )

# ==========================================================
# OVERSET CORRECTION
# ==========================================================

result = (
    OversetInterfaceCorrector2D
    .correct(
        basisL=basisL,
        uL=uL,
        basisR=basisR,
        uR=uR,
        interface_left=xI_left,
        interface_right=xI_right,
        interface_center = xI,
        alphas=alphas,
        omegas=omegas,
        direction="x",
    )
)

coefL = result.coefficient_left
coefR = result.coefficient_right

traceL = result.trace_left
traceR = result.trace_right

eL = result.correction_left
eR = result.correction_right

uLc = result.corrected_left
uRc = result.corrected_right

XL, YL = basisL.doflocs

XR, YR = basisR.doflocs

# ==========================================================
# CORRECTED SOLUTIONS
# ==========================================================

uLref=interp_full(XL,YL)
uRref=interp_full(XR,YR)
# true homogeneous components

uLh = uL - uLref
uRh = uR - uRref

# reconstruction errors of the homogeneous modes

hom_err_L = uLh - eL
hom_err_R = uRh - eR

# ==========================================================
# HOMOGENEOUS ERRORS
# ==========================================================

L2_hom_L = np.linalg.norm(hom_err_L)
L2_hom_R = np.linalg.norm(hom_err_R)

logger.info(
    f"L2(uLh-eL) = {L2_hom_L:.6e}"
)

logger.info(
    f"L2(uRh-eR) = {L2_hom_R:.6e}"
)


errL_before=np.linalg.norm(uL-uLref)
errL_after=np.linalg.norm(uLc-uLref)
errR_before=np.linalg.norm(uR-uRref)
errR_after=np.linalg.norm(uRc-uRref)

logger.info(f'Left before={errL_before:.6e}')
logger.info(f'Left after ={errL_after:.6e}')
logger.info(f'Right before={errR_before:.6e}')
logger.info(f'Right after ={errR_after:.6e}')

plt.figure(figsize=(10,8))
plt.tricontourf(XL,YL,eL,50)
plt.colorbar()
plt.title('Left homogeneous correction')
plt.savefig(run_dir/'left_correction.png',dpi=300)
plt.close()

plt.figure(figsize=(10,8))
plt.tricontourf(XR,YR,eR,50)
plt.colorbar()
plt.title('Right homogeneous correction')
plt.savefig(run_dir/'right_correction.png',dpi=300)
plt.close()


subdomains = [

    {
        "name": "A",
        "basis": basisL,
        "u": uL,
        "uc": uLc,
    },

    {
        "name": "B",
        "basis": basisR,
        "u": uR,
        "uc": uRc,
    },

]

assembly="sharp"  # sharp | blend"

logger.info(f"assembly {assembly}")

interp_unc = build_global_interpolator(
    subdomains,
    key="u",
    geometry="vertical",
    assembly=assembly,
    xI=xI,
    xI_left=xI_left,
    xI_right=xI_right,
)

interp_rec = build_global_interpolator(
    subdomains,
    key="uc",
    geometry="vertical",
    assembly=assembly,
    xI=xI,
    xI_left=xI_left,
    xI_right=xI_right,
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
err_after  = u_rec - uF

L2_before = np.linalg.norm(err_before)
L2_after  = np.linalg.norm(err_after)

logger.info(
    f"L2 before = {L2_before:.6e}"
)

logger.info(
    f"L2 after  = {L2_after:.6e}"
)

logger.info(
    f"Improvement = {L2_before/L2_after:.3f}"
)
# ==========================================================
# HOMOGENEOUS ERRORS
# ==========================================================

L2_hom_L = np.linalg.norm(hom_err_L)
L2_hom_R = np.linalg.norm(hom_err_R)

logger.info(
    f"L2(uLh-eL) = {L2_hom_L:.6e}"
)

logger.info(
    f"L2(uRh-eR) = {L2_hom_R:.6e}"
)

# ==========================================================
# REFERENCE SOLUTION
# ==========================================================

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    Xf,
    Yf,
    uF,
    cmap='viridis'
)

ax.set_title('Reference Solution')

plt.tight_layout()

plt.savefig(
    run_dir/'reference_surface_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    uF,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title('Reference Solution')

plt.tight_layout()

plt.savefig(
    run_dir/'reference_contour_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# RECONSTRUCTED SOLUTION
# ==========================================================

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    Xf,
    Yf,
    u_rec,
    cmap='viridis'
)

ax.set_title('Reconstructed Solution')

plt.tight_layout()

plt.savefig(
    run_dir/'reconstructed_surface_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    u_rec,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title('Reconstructed Solution')

plt.tight_layout()

plt.savefig(
    run_dir/'reconstructed_contour_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# ERROR BEFORE
# ==========================================================

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    Xf,
    Yf,
    err_before,
    cmap='seismic'
)

ax.set_title('Error Before Correction')

plt.tight_layout()

plt.savefig(
    run_dir/'error_before_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    err_before,
    levels=100,
    cmap='seismic'
)

plt.colorbar()

plt.title('Error Before Correction')

plt.tight_layout()

plt.savefig(
    run_dir/'error_before_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# ERROR AFTER
# ==========================================================

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    Xf,
    Yf,
    err_after,
    cmap='seismic'
)

ax.set_title('Error After Correction')

plt.tight_layout()

plt.savefig(
    run_dir/'error_after_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    Xf,
    Yf,
    err_after,
    levels=100,
    cmap='seismic'
)

plt.colorbar()

plt.title('Error After Correction')

plt.tight_layout()

plt.savefig(
    run_dir/'error_after_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# LEFT FEM HOMOGENEOUS
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XL,
    YL,
    uLh,
    cmap='viridis'
)

ax.set_title(
    r'Left FEM Homogeneous $u_h^{FEM}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_fem_homogeneous_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XL,
    YL,
    uLh,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title(
    r'Left FEM Homogeneous $u_h^{FEM}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_fem_homogeneous_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# LEFT ANALYTIC HOMOGENEOUS
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XL,
    YL,
    eL,
    cmap='viridis'
)

ax.set_title(
    r'Left Analytic Homogeneous $u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_analytic_homogeneous_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XL,
    YL,
    eL,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title(
    r'Left Analytic Homogeneous $u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_analytic_homogeneous_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# LEFT HOMOGENEOUS ERROR
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XL,
    YL,
    hom_err_L,
    cmap='seismic'
)

ax.set_title(
    r'Left Error $u_h^{FEM}-u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_homogeneous_error_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XL,
    YL,
    hom_err_L,
    levels=100,
    cmap='seismic'
)

plt.colorbar()

plt.title(
    r'Left Error $u_h^{FEM}-u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'left_homogeneous_error_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# RIGHT FEM HOMOGENEOUS
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XR,
    YR,
    uRh,
    cmap='viridis'
)

ax.set_title(
    r'Right FEM Homogeneous $u_h^{FEM}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_fem_homogeneous_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XR,
    YR,
    uRh,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title(
    r'Right FEM Homogeneous $u_h^{FEM}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_fem_homogeneous_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# RIGHT ANALYTIC HOMOGENEOUS
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XR,
    YR,
    eR,
    cmap='viridis'
)

ax.set_title(
    r'Right Analytic Homogeneous $u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_analytic_homogeneous_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XR,
    YR,
    eR,
    levels=100,
    cmap='viridis'
)

plt.colorbar()

plt.title(
    r'Right Analytic Homogeneous $u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_analytic_homogeneous_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# RIGHT HOMOGENEOUS ERROR
# ==========================================================

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(
    XR,
    YR,
    hom_err_R,
    cmap='seismic'
)

ax.set_title(
    r'Right Error $u_h^{FEM}-u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_homogeneous_error_3d.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(12,8))

plt.tricontourf(
    XR,
    YR,
    hom_err_R,
    levels=100,
    cmap='seismic'
)

plt.colorbar()

plt.title(
    r'Right Error $u_h^{FEM}-u_h^{analytic}$'
)

plt.tight_layout()

plt.savefig(
    run_dir/'right_homogeneous_error_2d.png',
    dpi=400
)

plt.close()

# ==========================================================
# EXPORT VALIDATION DATA
# ==========================================================

pd.DataFrame(
    {
        "x": XL,
        "y": YL,
        "uL_FEM_homogeneous": uLh,
        "uL_analytic_homogeneous": eL,
        "uL_error": hom_err_L
    }
).to_csv(
    run_dir/'left_homogeneous_validation.csv',
    index=False
)

pd.DataFrame(
    {
        "x": XR,
        "y": YR,
        "uR_FEM_homogeneous": uRh,
        "uR_analytic_homogeneous": eR,
        "uR_error": hom_err_R
    }
).to_csv(
    run_dir/'right_homogeneous_validation.csv',
    index=False
)

pd.DataFrame({
    'x':Xf,
    'y':Yf,
    'u_reference':uF,
    'u_uncorrected':u_unc,
    'u_reconstructed':u_rec,
    'error_before':err_before,
    'error_after':err_after
}).to_csv(
    run_dir/'full_validation_results.csv',
    index=False
)


print('Results stored in',run_dir)
