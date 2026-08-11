#!/usr/bin/env python3
"""
2D convergence study for exponential homogeneous reconstruction.

Runs the reconstruction for multiple mesh resolutions:
    nx_left = nx_right = ny = n
and measures
    u_h_FEM = u_FEM - u_exact
    error   = u_h_FEM - u_h_analytic

Produces:
- convergence_homogeneous.png
- convergence_total.png
- convergence_results.csv

Replace the body of run_validation(n) with your current reconstruction code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skfem import *
from skfem.helpers import dot, grad
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir=Path('runs')/f'convergence_{stamp}'
run_dir.mkdir(parents=True,exist_ok=True)

logger=logging.getLogger('convergence')
logger.setLevel(logging.INFO)
fh=logging.FileHandler(run_dir/'run.log')
logger.addHandler(fh)

mesh_sizes=[1, 0.1, 0.01]



def run_validation(dx):

    Lx = 100.0
    Ly = 40.0
    xI = 64.0
    nx_full=200
    ny_full=80
    L_left  = xI
    L_right = Lx - xI

    # same spatial spacing everywhere

    nx_left  = int(round(L_left  / dx))
    nx_right = int(round(L_right / dx))

    logger.info(
        f"dx={dx:.6e}, "
        f"nx_left={nx_left}, "
        f"nx_right={nx_right}, "
    )
    k_reaction=25.0

    F0=30.0
    P0=10.0
    P_amp=64.0
    kxP=0.64
    kyP=0.25

    A_f=1.0
    fx_wave=0.25
    fy_wave=0.15

    def F_func(x,y):
        #return F0+5*np.cos(0.064*x)*np.cos(0.05*y)
        return F0

    def Fx_func(x,y):
        #return -5*0.064*np.sin(0.064*x)*np.cos(0.05*y)
        return 0

    def P_func(x,y):
        return P0+P_amp*np.cos(kxP*x)*np.cos(kyP*y)

    def Px_func(x,y):
        return -P_amp*kxP*np.sin(kxP*x)*np.cos(kyP*y)

    def Py_func(x,y):
        return -P_amp*kyP*np.cos(kxP*x)*np.sin(kyP*y)

    def fx(x,y):
        #return A_f*fx_wave*np.cos(fx_wave*x)*np.sin(fy_wave*y)
        return 0

    def fy(x,y):
        #return A_f*fy_wave*np.sin(fx_wave*x)*np.cos(fy_wave*y)
        return 0

    def fxx(x,y):
        #return -A_f*(fx_wave**2)*np.sin(fx_wave*x)*np.sin(fy_wave*y)
        return 0

    def fyy(x,y):
        #return -A_f*(fy_wave**2)*np.sin(fx_wave*x)*np.sin(fy_wave*y)
        return 0

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
        
    def lambda_left(x, y):
        Fv  = F_func(x, y)
        Fxv = Fx_func(x, y)

        return (
            -Fxv
            + np.sqrt(Fxv**2 + 4*Fv*k_reaction)
        )/(2*Fv)    


    def left_mode(x, y, A):

        xx = np.linspace(xI, x, 200)

        lam = lambda_left(xx, y)

        phase = np.trapz(lam, xx)

        return A * np.exp(phase)
        
    def right_mode(x, y, B):

        xx = np.linspace(xI, x, 200)

        lam = lambda_left(xx, y)

        phase = np.trapz(lam, xx)

        return B * np.exp(-phase)


    @BilinearForm
    def diffusion(u,v,w):
        return F_func(w.x[0],w.x[1])*dot(grad(u),grad(v))

    @BilinearForm
    def reaction(u,v,w):
        return k_reaction*u*v

    @LinearForm
    def rhs(v,w):
        return forcing_term(w.x[0],w.x[1])*v


    def solve_rect(
        xmin,
        xmax,
        ymin,
        ymax,
        nx,
        ny,
        interface_bc=None,
        interface_side=None
    ):

        mesh = MeshTri.init_tensor(
            np.linspace(xmin, xmax, nx + 1),
            np.linspace(ymin, ymax, ny + 1)
        )

        basis = Basis(mesh, ElementTriP1())

        A = asm(diffusion, basis) + asm(reaction, basis)
        b = asm(rhs, basis)

        X = basis.doflocs[0]
        Y = basis.doflocs[1]

        tol = 1e-12

        left = basis.get_dofs(
            lambda x: np.isclose(x[0], xmin, atol=tol)
        )

        right = basis.get_dofs(
            lambda x: np.isclose(x[0], xmax, atol=tol)
        )

        bottom = basis.get_dofs(
            lambda x: np.isclose(x[1], ymin, atol=tol)
        )

        top = basis.get_dofs(
            lambda x: np.isclose(x[1], ymax, atol=tol)
        )

        D = np.unique(
            np.concatenate([
                np.asarray(left).ravel(),
                np.asarray(right).ravel(),
                np.asarray(bottom).ravel(),
                np.asarray(top).ravel()
            ])
        )

        xbc = basis.zeros()

        # -----------------------------
        # LEFT SIDE
        # -----------------------------

        if xmin == 0.0:

            for d in np.asarray(left).ravel():
                xbc[d] = bc_left(Y[d])

        elif interface_side == "left":

            for d in np.asarray(left).ravel():

                xbc[d] = np.interp(
                    Y[d],
                    y_interface,
                    interface_bc
                )

        # -----------------------------
        # RIGHT SIDE
        # -----------------------------

        if xmax == Lx:

            for d in np.asarray(right).ravel():
                xbc[d] = bc_right(Y[d])

        elif interface_side == "right":

            for d in np.asarray(right).ravel():

                xbc[d] = np.interp(
                    Y[d],
                    y_interface,
                    interface_bc
                )

        # -----------------------------
        # BOTTOM
        # -----------------------------

        for d in np.asarray(bottom).ravel():
            xbc[d] = bc_bottom(X[d])

        # -----------------------------
        # TOP
        # -----------------------------

        for d in np.asarray(top).ravel():
            xbc[d] = bc_top(X[d])

        AII, bI, xIvec, I = condense(
            A,
            b,
            D=D,
            x=xbc
        )

        u = xIvec.copy()
        u[I] = solve(AII, bI)

        return mesh, basis, u

    meshF,basisF,uF=solve_rect(0,Lx,0,Ly,nx_full,ny_full)
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
    ui_exact = interp_full(
        np.full_like(y_interface, xI),
        y_interface
    )
    rng = np.random.default_rng(42)


    noise_amplitude = 0.4

    ul_guess = (
        ui_exact
        + noise_amplitude *
        np.sin(2*np.pi*y_interface/Ly)
    )

    ur_guess = (
        ui_exact
        + noise_amplitude *
        np.sin(2*np.pi*y_interface/Ly)
    )

    meshL, basisL, uL = solve_rect(
        0,
        xI,
        0,
        Ly,
        nx_left,
        ny_full,
        interface_bc=ul_guess,
        interface_side="right"
    )
    meshR, basisR, uR = solve_rect(
        xI,
        Lx,
        0,
        Ly,
        nx_right,
        ny_full,
        interface_bc=ur_guess,
        interface_side="left"
    )


    Fv=F_func(xI,Ly/2)
    Fxv=Fx_func(xI,Ly/2)

    alpha=max(np.real(np.roots([ Fv,Fxv,-k_reaction ])))
    beta=max(np.real(np.roots([ -Fv,Fxv,k_reaction ])))
    
    lam = np.sqrt(
        mu**2 +
        k_reaction / F0
    )
    
    alpha = beta = lam 

    logger.info(f'alpha={alpha}')
    logger.info(f'beta={beta}')

    interp_full=LinearNDInterpolator(
        np.c_[basisF.doflocs[0],basisF.doflocs[1]],uF
    )

    xL_unique=np.unique(basisL.doflocs[0])
    xR_unique=np.unique(basisR.doflocs[0])

    x0=xL_unique[-3]
    x1=xR_unique[2]

    tol=1e-12
    idxL0=np.where(np.abs(basisL.doflocs[0]-x0)<tol)[0]
    idxR1=np.where(np.abs(basisR.doflocs[0]-x1)<tol)[0]

    yI=basisL.doflocs[1][idxL0]

    uL0_ref=interp_full(np.full_like(yI,x0),yI)
    uR1_ref=interp_full(np.full_like(yI,x1),yI)

    a0=np.exp(alpha*x0)
    a1=np.exp(alpha*x1)
    b0=np.exp(-beta*x0)
    b1=np.exp(-beta*x1)

    Aamp=np.zeros_like(yI)
    Bamp=np.zeros_like(yI)

    for j in range(len(yI)):
        M=np.array([
            [1,0,0,b0],
            [0,1,a1,0],
            [0,0,a0,0],
            [0,0,0,b1]
        ])

        rhs_vec=np.array([
            uL0_ref[j],
            uR1_ref[j],
            uL[idxL0[j]]-uL0_ref[j],
            uR[idxR1[j]]-uR1_ref[j]
        ])

        _,_,AA,BB=np.linalg.solve(M,rhs_vec)

        Aamp[j]=AA
        Bamp[j]=BB

    Afun=interp1d(yI,Aamp,fill_value='extrapolate')
    Bfun=interp1d(yI,Bamp,fill_value='extrapolate')

    XL,YL=basisL.doflocs
    XR,YR=basisR.doflocs

    eL=Afun(YL)*np.exp(alpha*XL)
    eR=Bfun(YR)*np.exp(-beta*XR)

    uLc=uL-eL
    uRc=uR-eR

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
    

    interpL_unc = LinearNDInterpolator(
        np.c_[XL,YL],
        uL
    )

    interpR_unc = LinearNDInterpolator(
        np.c_[XR,YR],
        uR
    )

    interpL_rec = LinearNDInterpolator(
        np.c_[XL,YL],
        uLc
    )

    interpR_rec = LinearNDInterpolator(
        np.c_[XR,YR],
        uRc
    )
    Xf = basisF.doflocs[0]
    Yf = basisF.doflocs[1]

    u_unc = np.zeros_like(uF)
    u_rec = np.zeros_like(uF)

    mask = Xf <= xI

    u_unc[mask] = interpL_unc(
        Xf[mask],
        Yf[mask]
    )

    u_unc[~mask] = interpR_unc(
        Xf[~mask],
        Yf[~mask]
    )

    u_rec[mask] = interpL_rec(
        Xf[mask],
        Yf[mask]
    )

    u_rec[~mask] = interpR_rec(
        Xf[~mask],
        Yf[~mask]
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
    # ANALYTICAL HOMOGENEOUS SOLUTION
    # ==========================================================

    mu = 2.0 * np.pi / Ly

    lambda_exact = np.sqrt(
        mu**2 + k_reaction / F0
    )

    XL, YL = basisL.doflocs
    XR, YR = basisR.doflocs

    # ==========================================================
    # EXACT HOMOGENEOUS SOLUTION
    # ==========================================================

    mu = 2.0 * np.pi / Ly

    lam = np.sqrt(
        mu**2 +
        k_reaction / F0
    )

    A0 = noise_amplitude

    uL_exact_h = (
        A0
        * np.sin(mu * YL)
        * np.sinh(lam * XL)
        / np.sinh(lam * xI)
    )

    uR_exact_h = (
        -A0
        * np.sin(mu * YR)
        * np.sinh(lam * (Lx - XR))
        / np.sinh(lam * (Lx - xI))
    )

    hom_err_L = uLh - uL_exact_h
    hom_err_R = uRh - uR_exact_h

    L2_hom_L = np.linalg.norm(hom_err_L)
    L2_hom_R = np.linalg.norm(hom_err_R)

    logger.info(
        f"L2(FEM_h-analytic_h)_L = "
        f"{L2_hom_L:.6e}"
    )

    logger.info(
        f"L2(FEM_h-analytic_h)_R = "
        f"{L2_hom_R:.6e}"
    )
    
    y_test = y_interface

    uh_left_interface = interp1d(
        YL[np.isclose(XL, xI)],
        uLh[np.isclose(XL, xI)],
        fill_value="extrapolate"
    )(y_test)

    analytic_interface = (
        noise_amplitude
        * np.sin(2*np.pi*y_test/Ly)
    )

    logger.info(
        np.linalg.norm(
            uh_left_interface
            - analytic_interface
        )
    )
    
    interface_x = np.max(XL)

    uh_left_interface = interp1d(
        YL[np.isclose(XL, interface_x)],
        uLh[np.isclose(XL, interface_x)],
        fill_value="extrapolate"
    )(y_interface)

    analytic_interface = (
        noise_amplitude
        * np.sin(2*np.pi*y_interface/Ly)
    )

    logger.info(
        f"Interface error = "
        f"{np.linalg.norm(uh_left_interface - analytic_interface):.6e}"
    )
    
    interface_x = np.min(XR)

    uh_right_interface = interp1d(
        YR[np.isclose(XR, interface_x)],
        uRh[np.isclose(XR, interface_x)],
        fill_value="extrapolate"
    )(y_interface)

    analytic_interface = (
        -noise_amplitude
        * np.sin(2*np.pi*y_interface/Ly)
    )

    logger.info(
        f"Right interface error = "
        f"{np.linalg.norm(uh_right_interface - analytic_interface):.6e}"
    )
            
    return {
        'dx':dx,
        'L2_hom_L':L2_hom_L,
        'L2_hom_R':L2_hom_R,
        'L2_total_before':L2_before,
        'L2_total_after':L2_after
    }


results=[]

for n in mesh_sizes:

    logger.info(f'Running n={n}')

    results.append(
        run_validation(n)
    )

conv_df=pd.DataFrame(results)

N=conv_df['dx'].values
err_hom_L=conv_df['L2_hom_L'].values
err_hom_R=conv_df['L2_hom_R'].values
err_before=conv_df['L2_total_before'].values
err_after=conv_df['L2_total_after'].values


plt.figure(figsize=(10,8))

plt.loglog(
    N,
    err_hom_L,
    'o-',
    lw=3,
    label='Left homogeneous'
)

plt.loglog(
    N,
    err_hom_R,
    's-',
    lw=3,
    label='Right homogeneous'
)

plt.grid(True,which='both')
plt.xlabel('dx')
plt.ylabel(r'||u_h^FEM-u_h^analytic||')
plt.title('Homogeneous Approximation Error')
plt.legend()

plt.savefig(
    run_dir/'convergence_homogeneous.png',
    dpi=400
)

plt.close()


plt.figure(figsize=(10,8))

plt.loglog(
    N,
    err_before,
    'o-',
    lw=3,
    label='Before correction'
)

plt.loglog(
    N,
    err_after,
    's-',
    lw=3,
    label='After correction'
)

plt.grid(True,which='both')
plt.xlabel('dx')
plt.ylabel('L2 error')
plt.title('Total Reconstruction Error')
plt.legend()

plt.savefig(
    run_dir/'convergence_total.png',
    dpi=400
)

plt.close()


order_left=-np.polyfit(np.log(N),np.log(err_hom_L),1)[0]
order_right=-np.polyfit(np.log(N),np.log(err_hom_R),1)[0]

logger.info(f'Estimated left order  = {order_left:.4f}')
logger.info(f'Estimated right order = {order_right:.4f}')

conv_df.to_csv(
    run_dir/'convergence_results.csv',
    index=False
)

print('Results stored in',run_dir)
