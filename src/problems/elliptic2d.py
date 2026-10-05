"""
elliptic2d.py
=============

Two-dimensional elliptic model problem used throughout the overlap
reconstruction framework.

Governing equation

    k*u - div(h grad(u))
        =
        div(p grad(f))

Author
------
Lorenzo Zambelli
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skfem import BilinearForm, LinearForm
from skfem.helpers import dot, grad


@dataclass(slots=True)
class Elliptic2DCoefficients:
    """
    Coefficient container.

    Inputs:
        k : float
        h0 : float
        p0 : float
        p_amp : float
        kxP : float
        kyP : float
        A_f : float
        fx_wave : float
        fy_wave : float
    """

    k: float = 25.0

    h0: float = 30.0

    p0: float = 10.0
    p_amp: float = 64.0

    kxP: float = 0.64
    kyP: float = 0.25

    A_f: float = 1.0

    fx_wave: float = 0.25
    fy_wave: float = 0.15


class Elliptic2DProblem:
    """
    Two-dimensional elliptic boundary-value problem.

    Inputs:
        coeffs : Elliptic2DCoefficients

    Outputs:
        Elliptic2DProblem
    """

    def __init__(
        self,
        coeffs: Elliptic2DCoefficients | None = None,
    ):

        self.c = (
            coeffs
            if coeffs is not None
            else Elliptic2DCoefficients()
        )

        self.k = self.c.k

    def h(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            h(x,y)
        """

        return self.c.h0

    def hx(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            h_x(x,y)
        """

        return 0.0

    def hy(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            h_y(x,y)
        """

        return 0.0

    def p(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            p(x,y)
        """

        c=self.c

        return (
            c.p0
            +
            c.p_amp
            *
            np.cos(c.kxP*x)
            *
            np.cos(c.kyP*y)
        )

    def px(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            p_x(x,y)
        """

        c=self.c

        return (
            -c.p_amp
            *
            c.kxP
            *
            np.sin(c.kxP*x)
            *
            np.cos(c.kyP*y)
        )

    def py(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            p_y(x,y)
        """

        c=self.c

        return (
            -c.p_amp
            *
            c.kyP
            *
            np.cos(c.kxP*x)
            *
            np.sin(c.kyP*y)
        )

    def fx(self,x,y,t=0.0):
        """
        Inputs:
            x,y,t

        Outputs:
            f_x(x,y,t)
        """

        c=self.c

        return (
            c.A_f
            *
            c.fx_wave
            *
            np.cos(c.fx_wave*x)
            *
            np.sin(c.fy_wave*y)
        )

    def fy(self,x,y,t=0.0):
        """
        Inputs:
            x,y,t

        Outputs:
            f_y(x,y,t)
        """

        c=self.c

        return (
            c.A_f
            *
            c.fy_wave
            *
            np.sin(c.fx_wave*x)
            *
            np.cos(c.fy_wave*y)
        )

    def fxx(self,x,y,t=0.0):
        """
        Inputs:
            x,y,t

        Outputs:
            f_xx(x,y,t)
        """

        c=self.c

        return (
            -c.A_f
            *
            c.fx_wave**2
            *
            np.sin(c.fx_wave*x)
            *
            np.sin(c.fy_wave*y)
        )

    def fyy(self,x,y,t=0.0):
        """
        Inputs:
            x,y,t

        Outputs:
            f_yy(x,y,t)
        """

        c=self.c

        return (
            -c.A_f
            *
            c.fy_wave**2
            *
            np.sin(c.fx_wave*x)
            *
            np.sin(c.fy_wave*y)
        )

    def forcing(self,x,y,t=0.0):
        """
        Inputs:
            x,y,t

        Outputs:
            div(p grad(f))
        """

        return (
            self.px(x,y)*self.fx(x,y,t)
            +
            self.py(x,y)*self.fy(x,y,t)
            +
            self.p(x,y)
            *
            (
                self.fxx(x,y,t)
                +
                self.fyy(x,y,t)
            )
        )

    def forms(self):
        """
        Outputs:
            diffusion,reaction,rhs
        """

        @BilinearForm
        def diffusion(u,v,w):

            return (
                self.h(
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

            return self.k*u*v

        @LinearForm
        def rhs(v,w):

            return (
                self.forcing(
                    w.x[0],
                    w.x[1],
                    0.0,
                )
                *
                v
            )

        return (
            diffusion,
            reaction,
            rhs,
        )

    def decay_parameter(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            lambda
        """

        return np.sqrt(
            self.k
            /
            self.h(x,y)
        )

    def boundary_layer_thickness(self,x,y):
        """
        Inputs:
            x,y

        Outputs:
            delta
        """

        return 1.0/self.decay_parameter(x,y)

    def modal_decay_x(self,n,Ly,h_value):
        """
        Inputs:
            n,Ly,h_value

        Outputs:
            alpha_n
        """

        beta=n*np.pi/Ly

        return np.sqrt(
            self.k/h_value
            +
            beta**2
        )

    def modal_decay_y(self,n,Lx,h_value):
        """
        Inputs:
            n,Lx,h_value

        Outputs:
            gamma_n
        """

        beta=n*np.pi/Lx

        return np.sqrt(
            self.k/h_value
            +
            beta**2
        )

    def average_h(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        nx=50,
        ny=50,
    ):
        """
        Inputs:
            xmin,xmax,ymin,ymax,nx,ny

        Outputs:
            h_avg
        """

        xx=np.linspace(xmin,xmax,nx)
        yy=np.linspace(ymin,ymax,ny)

        X,Y=np.meshgrid(xx,yy)

        return np.mean(self.h(X,Y))

    def mu(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
    ):
        """
        Inputs:
            xmin,xmax,ymin,ymax

        Outputs:
            mu
        """

        return np.sqrt(
            self.k
            /
            self.average_h(
                xmin,
                xmax,
                ymin,
                ymax,
            )
        )

    def summary(self):
        """
        Outputs:
            dict
        """

        return {
            "dimension": 2,
            "equation":
            "k*u-div(h grad(u))=div(p grad(f))",
            "k": self.k,
        }