"""
elliptic1d.py
=============

Author
------
Lorenzo Zambelli

Version
-------
1.0

Description
-----------
Definition of the one-dimensional elliptic boundary-value problem used
throughout the overlap-reconstruction framework.

The module provides a lightweight representation of the physical
coefficients, forcing terms, and analytical quantities associated with
the one-dimensional reduction of the Variational Boussinesq Model (VBM)
elliptic equation.

Mathematical Model
------------------
The governing equation is

    k*u - d/dx(h*u_x)
        =
        d/dx(p*f_x)

or equivalently

    k*u - d/dx(h*u_x)
        =
        p_x*f_x + p*f_xx

where

    u(x,t) : unknown field
    h(x,t) : water-depth coefficient
    p(x,t) : pressure-related coefficient
    f(x,t) : free-surface-related quantity
    k      : reaction coefficient

For the homogeneous problem

    k*u - d/dx(h*u_x) = 0,

and under the assumption that h is locally constant,
the characteristic decay parameter is

    λ = sqrt(k / h)

and the corresponding boundary-layer thickness is

    δ = 1 / λ.

Responsibilities
----------------
* Store physical coefficients.
* Evaluate forcing terms.
* Compute local decay parameters.
* Compute characteristic boundary-layer lengths.
* Provide utilities shared by numerical solvers and
  reconstruction algorithms.

Examples
--------
problem = Elliptic1DProblem(
    k=25.0,
    h=h_function,
    h_x=h_prime,
    p=p_function,
    p_x=p_prime,
    f_x=f_x_function,
    f_xx=f_xx_function
)

rhs = problem.forcing(x, t)

lam = problem.decay_parameter(x)

delta = problem.boundary_layer_thickness(x)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Elliptic1DProblem:
    """
    One-dimensional elliptic problem corresponding to

        k*u - d/dx(h*u_x)
            =
            d/dx(p*f_x).

    Parameters
    ----------
    k : float
        Reaction coefficient.

    h : callable
        Water-depth coefficient h(x).

    h_x : callable
        Spatial derivative of h(x).

    p : callable
        Pressure-related coefficient p(x).

    p_x : callable
        Spatial derivative of p(x).

    f_x : callable
        First spatial derivative of f(x,t).

    f_xx : callable
        Second spatial derivative of f(x,t).

    Notes
    -----
    The callables are assumed to operate on NumPy arrays.
    """

    k: float
    h: callable
    h_x: callable
    p: callable
    p_x: callable
    f_x: callable
    f_xx: callable

    def forcing(self, x, t):
        """
        Evaluate the forcing term

            d/dx(p*f_x).

        Using the product rule,

            d/dx(p*f_x)
            =
            p_x*f_x + p*f_xx.

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        t : float
            Physical time.

        Returns
        -------
        float or ndarray
            Value of the forcing term at the specified
            spatial locations.
        """

        return (
            self.p_x(x) * self.f_x(x, t)
            +
            self.p(x) * self.f_xx(x, t)
        )

    def local_constant_coefficient(self, x):
        """
        Evaluate the local coefficient h(x).

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        Returns
        -------
        float or ndarray
            Value of h at x.
        """

        return self.h(x)

    def decay_parameter(self, x):
        """
        Compute the local homogeneous decay parameter.

        Assuming h varies slowly in the neighborhood
        of x, the characteristic equation

            h*λ² - k = 0

        yields

            λ = sqrt(k/h).

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        Returns
        -------
        float or ndarray
            Local decay parameter λ.
        """

        return np.sqrt(
            self.k / self.h(x)
        )

    def boundary_layer_thickness(self, x):
        """
        Compute the characteristic boundary-layer thickness.

        The homogeneous correction decays according to

            exp(-λx),

        where λ is the local decay parameter.
        The corresponding decay length is

            δ = 1 / λ.

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        Returns
        -------
        float or ndarray
            Characteristic decay length δ.
        """

        return (
            1.0 /
            self.decay_parameter(x)
        )

    def characteristic_roots(self, x):
        """
        Compute the roots of the local characteristic equation.

        For locally constant h,

            h*λ² - k = 0,

        giving

            λ₁ = +sqrt(k/h)
            λ₂ = -sqrt(k/h).

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        Returns
        -------
        tuple
            (positive_root, negative_root)
        """

        lam = self.decay_parameter(x)

        return (
            lam,
            -lam,
        )

    def local_characteristic_roots(self, x):
        """
        Compute characteristic roots for smoothly varying h(x).

        Starting from

            -(h*u_x)_x + k*u = 0,

        and expanding,

            -h*u_xx - h_x*u_x + k*u = 0.

        Seeking solutions of the form

            u = exp(λx)

        gives the local characteristic equation

            h*λ² + h_x*λ - k = 0.

        Parameters
        ----------
        x : float or ndarray
            Spatial coordinate(s).

        Returns
        -------
        tuple
            Positive and negative roots of the local
            characteristic equation.
        """

        h_value = self.h(x)
        hx_value = self.h_x(x)

        discriminant = (
            hx_value**2
            +
            4.0 * h_value * self.k
        )

        root_plus = (
            -hx_value
            +
            np.sqrt(discriminant)
        ) / (2.0 * h_value)

        root_minus = (
            -hx_value
            -
            np.sqrt(discriminant)
        ) / (2.0 * h_value)

        return (
            root_plus,
            root_minus,
        )

    def averaged_decay_parameter(
        self,
        x_start,
        x_end,
        n_samples=100,
    ):
        """
        Compute an average decay parameter over
        an interval.

        This utility is useful inside overlap
        regions where h varies slowly and a
        single representative λ is desired.

        Parameters
        ----------
        x_start : float
            Beginning of interval.

        x_end : float
            End of interval.

        n_samples : int, optional
            Number of sampling points.

        Returns
        -------
        float
            Average decay parameter on
            [x_start, x_end].
        """

        x = np.linspace(
            x_start,
            x_end,
            n_samples,
        )

        lam = self.decay_parameter(x)

        return np.mean(lam)

    def summary(self):
        """
        Return a dictionary describing the problem.

        Returns
        -------
        dict
            Problem metadata.
        """

        return {
            "reaction_coefficient": self.k,
            "equation":
                "k*u - d/dx(h*u_x) = d/dx(p*f_x)",
            "dimension": 1,
        }