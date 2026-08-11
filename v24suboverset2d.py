#!/usr/bin/env python3
"""
2D hierarchical four-subdomain validation

           D3 | D4
          ---------
           D1 | D2

The four subdomains are solved independently using perturbed
interface data. 
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
)

from utils.overset_interface_corrector_rewritten import (
    compute_modes,
    OversetInterfaceCorrector2D,
)

from utils.overset_hierarchical_xy_corrector import (
    OversetHierarchicalXYCorrector,
    FourDomainLCorrector
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

xI = Lx / 2.0
yI = Ly / 2.0
xmid = xI
ymid = yI
# ==========================================================
# MESH RESOLUTION
# ==========================================================

nx_full = 300
ny_full = 300

nx_sub = nx_full // 2
ny_sub = ny_full // 2

# ==========================================================
# OVERSET OVERLAPS
# ==========================================================

overlap_x_left = 5
overlap_x_right = 5

overlap_y_top = 5
overlap_y_bottom = 5

hx_left = xI / nx_sub
hx_right = (Lx - xI) / nx_sub

hy_bottom = yI / ny_sub
hy_top = (Ly - yI) / ny_sub

xI_left = (
    xI
    +
    overlap_x_left * hx_left
)

xI_right = (
    xI
    -
    overlap_x_right * hx_right
)

yI_top = (
    yI
    -
    overlap_y_top * hy_top
)

yI_bottom = (
    yI
    +
    overlap_y_bottom * hy_bottom
)

# ==========================================================
# PDE PARAMETERS
# ==========================================================

k_reaction=25.0

F0=10
P0=10
P_amp=64.0
kxP=0.64
kyP=0.25

A_f=2.0
fx_wave=0.25
fy_wave=0.15

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
# BOUNDARY COORDINATES
# ==========================================================

y_bottom = np.linspace(
    0.0,
    yI_bottom,
    ny_sub + 1,
)

y_top = np.linspace(
    yI_top,
    Ly,
    ny_sub + 1,
)

x_left = np.linspace(
    0.0,
    xI_left,
    nx_sub + 1,
)

x_right = np.linspace(
    xI_right,
    Lx,
    nx_sub + 1,
)

# ==========================================================
# EXACT TRACES ON OVERSET BOUNDARIES
# ==========================================================

# vertical interfaces

u12_left_exact = interp_full(
    np.full_like(y_bottom, xI_left),
    y_bottom,
)

u12_right_exact = interp_full(
    np.full_like(y_bottom, xI_right),
    y_bottom,
)

u34_left_exact = interp_full(
    np.full_like(y_top, xI_left),
    y_top,
)

u34_right_exact = interp_full(
    np.full_like(y_top, xI_right),
    y_top,
)

# horizontal interfaces

u1top_exact = interp_full(
    x_left,
    np.full_like(
        x_left,
        yI_bottom,
    ),
)

u2top_exact = interp_full(
    x_right,
    np.full_like(
        x_right,
        yI_bottom,
    ),
)

u3bottom_exact = interp_full(
    x_left,
    np.full_like(
        x_left,
        yI_top,
    ),
)

u4bottom_exact = interp_full(
    x_right,
    np.full_like(
        x_right,
        yI_top,
    ),
)

eps = 30

u12_left = (
    1.0
    + eps*np.sin(np.pi*y_bottom/yI_bottom)
) * u12_left_exact

u12_right = (
    1.0
    - eps*np.sin(np.pi*y_bottom/yI_bottom)
) * u12_right_exact

u34_left = (
    1.0
    + eps*np.sin(
        np.pi*(y_top-yI_top)/(Ly-yI_top)
    )
) * u34_left_exact

u34_right = (
    1.0
    - eps*np.sin(
        np.pi*(y_top-yI_top)/(Ly-yI_top)
    )
) * u34_right_exact

u1_top = (
    1.0
    + eps*np.sin(np.pi*x_left/xI_left)
) * u1top_exact

u2_top = (
    1.0
    - eps*np.sin(
        np.pi*(x_right-xI_right)/(Lx-xI_right)
    )
) * u2top_exact

u3_bottom = (
    1.0
    + eps*np.sin(np.pi*x_left/xI_left)
) * u3bottom_exact

u4_bottom = (
    1.0
    - eps*np.sin(
        np.pi*(x_right-xI_right)/(Lx-xI_right)
    )
) * u4bottom_exact
# ==========================================================
# DOMAIN D1
# ==========================================================

logger.info("Solving D1")

bc1 = {
    "left": lambda x, y: bc_left(y),

    "bottom": lambda x, y: bc_bottom(x),

    "right": {
        "coords": y_bottom,
        "values": u12_left,
    },

    "top": {
        "coords": x_left,
        "values": u1_top,
    },
}
mesh1, basis1, u1 = solve_rect(

    xmin=0.0,
    xmax=xI_left,

    ymin=0.0,
    ymax=yI_bottom,

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
    "left": {
        "coords": y_bottom,
        "values": u12_right,
    },

    "right": lambda x, y: bc_right(y),

    "bottom": lambda x, y: bc_bottom(x),

    "top": {
        "coords": x_right,
        "values": u2_top,
    },
}
mesh2, basis2, u2 = solve_rect(

    xmin=xI_right,
    xmax=Lx,

    ymin=0.0,
    ymax=yI_bottom,

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
    "left": lambda x, y: bc_left(y),

    "right": {
        "coords": y_top,
        "values": u34_left,
    },

    "bottom": {
        "coords": x_left,
        "values": u3_bottom,
    },

    "top": lambda x, y: bc_top(x),
}

mesh3, basis3, u3 = solve_rect(

    xmin=0.0,
    xmax=xI_left,

    ymin=yI_top,
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
    "left": {
        "coords": y_top,
        "values": u34_right,
    },

    "right": lambda x, y: bc_right(y),

    "bottom": {
        "coords": x_right,
        "values": u4_bottom,
    },

    "top": lambda x, y: bc_top(x),
}

mesh4, basis4, u4 = solve_rect(

    xmin=xI_right,
    xmax=Lx,

    ymin=yI_top,
    ymax=Ly,

    nx=nx_sub,
    ny=ny_sub,

    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,

    boundary_conditions=bc4,
)

# ==========================================================
# INTERFACE DIFFUSION AVERAGES
# ==========================================================

ys_bottom = np.linspace(
    0.0,
    ymid,
    400,
)

ys_top = np.linspace(
    ymid,
    Ly,
    400,
)

xs_left = np.linspace(
    0.0,
    xmid,
    400,
)

xs_right = np.linspace(
    xmid,
    Lx,
    400,
)

F_vertical_bottom = np.mean(
    F_func(
        np.full_like(
            ys_bottom,
            xmid,
        ),
        ys_bottom,
    )
)

F_vertical_top = np.mean(
    F_func(
        np.full_like(
            ys_top,
            xmid,
        ),
        ys_top,
    )
)

F_horizontal_left = np.mean(
    F_func(
        xs_left,
        np.full_like(
            xs_left,
            ymid,
        ),
    )
)

F_horizontal_right = np.mean(
    F_func(
        xs_right,
        np.full_like(
            xs_right,
            ymid,
        ),
    )
)

# ==========================================================
# MODAL LENGTHS
# ==========================================================

# --------------------------------------------------
# domain extents
# --------------------------------------------------

L_D1_y = yI_bottom
L_D2_y = yI_bottom

L_D3_y = Ly - yI_top
L_D4_y = Ly - yI_top

L_D1_x = xI_left
L_D3_x = xI_left

L_D2_x = Lx - xI_right
L_D4_x = Lx - xI_right

# ==========================================================
# VERTICAL MODES
# ==========================================================
modes_x = 20
modes_y = 20

alphas_vertical_bottom, omegas_vertical_bottom = (
    compute_modes(
        Ly=L_D1_y,
        k_reaction=k_reaction,
        F_interface=F_vertical_bottom,
        n_modes=modes_y,
    )
)

alphas_vertical_top, omegas_vertical_top = (
    compute_modes(
        Ly=L_D3_y,
        k_reaction=k_reaction,
        F_interface=F_vertical_top,
        n_modes=modes_y,
    )
)

# ==========================================================
# HORIZONTAL MODES
# ==========================================================

alphas_horizontal_left, omegas_horizontal_left = (
    compute_modes(
        Ly=L_D1_x,
        k_reaction=k_reaction,
        F_interface=F_horizontal_left,
        n_modes=modes_x,
    )
)

alphas_horizontal_right, omegas_horizontal_right = (
    compute_modes(
        Ly=L_D2_x,
        k_reaction=k_reaction,
        F_interface=F_horizontal_right,
        n_modes=modes_x,
    )
)

logger.info(
    f"F_vertical_bottom = "
    f"{F_vertical_bottom:.6e}"
)

logger.info(
    f"F_vertical_top = "
    f"{F_vertical_top:.6e}"
)

logger.info(
    f"F_horizontal_left = "
    f"{F_horizontal_left:.6e}"
)

logger.info(
    f"F_horizontal_right = "
    f"{F_horizontal_right:.6e}"
)



# ==========================================================
# CROSS INTERFACE OVERSSET RECONSTRUCTION
# ==========================================================

# ==========================================================
# NEIGHBOR INTERPOLATORS
# ==========================================================

interp_D1 = LinearNDInterpolator(
    np.c_[
        basis1.doflocs[0],
        basis1.doflocs[1],
    ],
    u1,
)

interp_D2 = LinearNDInterpolator(
    np.c_[
        basis2.doflocs[0],
        basis2.doflocs[1],
    ],
    u2,
)

interp_D3 = LinearNDInterpolator(
    np.c_[
        basis3.doflocs[0],
        basis3.doflocs[1],
    ],
    u3,
)

interp_D4 = LinearNDInterpolator(
    np.c_[
        basis4.doflocs[0],
        basis4.doflocs[1],
    ],
    u4,
)


res12 = (
    OversetInterfaceCorrector2D.correct(
        basis1,
        u1,
        basis2,
        u2,
        xI_left,
        xI_right,
        xI,
        alphas_vertical_bottom,
        omegas_vertical_bottom,
        direction="x",
    )
)

u1v = res12.corrected_left
u2v = res12.corrected_right

res34 = (
    OversetInterfaceCorrector2D.correct(
        basis3,
        u3,
        basis4,
        u4,
        xI_left,
        xI_right,
        xI,
        alphas_vertical_top,
        omegas_vertical_top,
        direction="x",
    )
)

u3v = res34.corrected_left
u4v = res34.corrected_right


interp_D1v = LinearNDInterpolator(
    np.c_[basis1.doflocs[0], basis1.doflocs[1]],
    u1v,
)

interp_D2v = LinearNDInterpolator(
    np.c_[basis2.doflocs[0], basis2.doflocs[1]],
    u2v,
)

interp_D3v = LinearNDInterpolator(
    np.c_[basis3.doflocs[0], basis3.doflocs[1]],
    u3v,
)

interp_D4v = LinearNDInterpolator(
    np.c_[basis4.doflocs[0], basis4.doflocs[1]],
    u4v,
)

res13 = (
    OversetInterfaceCorrector2D.correct(
        basis1,
        u1v,
        basis3,
        u3v,
        yI_bottom,
        yI_top,
        yI,
        alphas_horizontal_left,
        omegas_horizontal_left,
        direction="y",
    )
)

res24 = (
    OversetInterfaceCorrector2D.correct(
        basis2,
        u2v,
        basis4,
        u4v,
        yI_bottom,
        yI_top,
        yI,
        alphas_horizontal_right,
        omegas_horizontal_right,
        direction="y",
    )
)

u1y = res13.corrected_left
u3y = res13.corrected_right

u2y = res24.corrected_left
u4y = res24.corrected_right

hierarchy = FourDomainLCorrector.apply(
    u1,
    u2,
    u3,
    u4,
    res12,
    res34,
    res13,
    res24,
)

D1 = hierarchy["D1"]
D2 = hierarchy["D2"]
D3 = hierarchy["D3"]
D4 = hierarchy["D4"]

u1f = D1.corrected
u2f = D2.corrected
u3f = D3.corrected
u4f = D4.corrected

corr1 = D1.correction_total
corr2 = D2.correction_total
corr3 = D3.correction_total
corr4 = D4.correction_total

corrxy1 = D1.correction_xy
corrxy2 = D2.correction_xy
corrxy3 = D3.correction_xy
corrxy4 = D4.correction_xy
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
# ==========================================================
# GLOBAL RECONSTRUCTION
# ==========================================================
interpD1 = LinearNDInterpolator(
    np.c_[basis1.doflocs[0], basis1.doflocs[1]],
    u1f,
)

interpD2 = LinearNDInterpolator(
    np.c_[basis2.doflocs[0], basis2.doflocs[1]],
    u2f,
)

interpD3 = LinearNDInterpolator(
    np.c_[basis3.doflocs[0], basis3.doflocs[1]],
    u3f,
)

interpD4 = LinearNDInterpolator(
    np.c_[basis4.doflocs[0], basis4.doflocs[1]],
    u4f,
)
assembly = "blend"      # sharp | blend

logger.info(
    f"assembly = {assembly}"
)

interp_unc = build_global_interpolator(
    subdomains,
    key="u",
    geometry="cross",
    assembly=assembly,

    xI=xI,
    yI=yI,

    xI_left=xI_left,
    xI_right=xI_right,

    yI_top=yI_top,
    yI_bottom=yI_bottom,
)

interp_rec = build_global_interpolator(
    subdomains,
    key="uc",
    geometry="cross",
    assembly=assembly,

    xI=xI,
    yI=yI,

    xI_left=xI_left,
    xI_right=xI_right,

    yI_top=yI_top,
    yI_bottom=yI_bottom,
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

for name, res in [

    ("D12", res12),
    ("D34", res34),
    ("D13", res13),
    ("D24", res24),

]:

    logger.info(
        f"{name} ||coef_left||="
        f"{np.linalg.norm(res.coefficient_left):.6e}"
    )

    logger.info(
        f"{name} ||coef_right||="
        f"{np.linalg.norm(res.coefficient_right):.6e}"
    )
    
for name, corrxy in [

    ("D1", corrxy1),
    ("D2", corrxy2),
    ("D3", corrxy3),
    ("D4", corrxy4),

]:

    logger.info(
        f"{name} ||corr_xy|| = "
        f"{np.linalg.norm(corrxy):.6e}"
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