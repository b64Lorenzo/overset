"""
test_fem1d_validation.py
========================

Validation of

    src.problems.elliptic1d
    src.solver.fem1d

Manufactured solution:

    u(x)=sin(pi x)

on

    x in [0,1]

satisfying

    k*u - u_xx = rhs

with homogeneous Dirichlet conditions.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.utils.logger import Logger

from src.problems.elliptic1d import (
    Elliptic1DProblem,
)

from src.solver.fem1d import (
    FEMEllipticSolver1D,
)


# ============================================================
# LOGGER
# ============================================================

log_mgr = Logger(
    name="test_fem1d_validation",
    run_prefix="test_fem1d_validation",
)

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()


# ============================================================
# MANUFACTURED PROBLEM
# ============================================================

k_reaction = 25.0


def h(x):
    return np.ones_like(np.asarray(x))


def h_x(x):
    return np.zeros_like(np.asarray(x))


#
# IMPORTANT
#
# forcing = p_x*f_x + p*f_xx
#
# choose
#
# p = 1
# p_x = 0
#
# so that
#
# forcing = f_xx
#

def p(x):
    return np.ones_like(np.asarray(x))


def p_x(x):
    return np.zeros_like(np.asarray(x))


# ============================================================
# EXACT SOLUTION
# ============================================================

def u_exact(x):
    return np.sin(np.pi * x)


def forcing_exact(x):

    return (
        k_reaction
        + np.pi**2
    ) * np.sin(np.pi * x)


def f_x(x, t):
    return np.zeros_like(np.asarray(x))


def f_xx(x, t):
    return forcing_exact(x)


# ============================================================
# BUILD PROBLEM
# ============================================================

problem = Elliptic1DProblem(
    k=k_reaction,
    h=h,
    h_x=h_x,
    p=p,
    p_x=p_x,
    f_x=f_x,
    f_xx=f_xx,
)

solver = FEMEllipticSolver1D(problem)


# ============================================================
# PROBLEM DIAGNOSTICS
# ============================================================

logger.info(problem.summary())

x_test = 0.5

lam = problem.decay_parameter(x_test)

delta = problem.boundary_layer_thickness(
    x_test
)

root_plus, root_minus = (
    problem.characteristic_roots(
        x_test
    )
)

logger.info(
    "Decay parameter lambda = %.6f",
    lam,
)

logger.info(
    "Boundary layer thickness delta = %.6f",
    delta,
)

logger.info(
    "Characteristic roots = (%f,%f)",
    root_plus,
    root_minus,
)

assert np.isclose(
    lam,
    np.sqrt(k_reaction),
)

assert np.isclose(
    delta,
    1.0 / np.sqrt(k_reaction),
)


# ============================================================
# CONVERGENCE STUDY
# ============================================================

mesh_sizes = [
    20,
    40,
    80,
    160,
    320,
    640,
]

errors = []

for n in mesh_sizes:

    x, u = solver.solve(
        a=0.0,
        b=1.0,
        n_elements=n,
        bc_left=0.0,
        bc_right=0.0,
    )

    u_ref = u_exact(x)

    error = np.sqrt(
        np.mean(
            (u - u_ref)**2
        )
    )

    errors.append(error)

    logger.info(
        "N=%4d  error=%.12e",
        n,
        error,
    )


# ============================================================
# OBSERVED ORDER
# ============================================================

orders = []

for i in range(
    1,
    len(errors),
):

    p_order = (
        np.log(
            errors[i - 1]
            /
            errors[i]
        )
        /
        np.log(2.0)
    )

    orders.append(
        p_order
    )

    logger.info(
        "order[%d]=%.6f",
        i,
        p_order,
    )

average_order = np.mean(
    orders
)

logger.info(
    "Average order = %.6f",
    average_order,
)

#
# linear finite elements
# nodal error should approach 2
#

assert average_order > 1.8


# ============================================================
# REFERENCE SOLUTION
# ============================================================

x_ref, u_ref = solver.solve(
    a=0.0,
    b=1.0,
    n_elements=1000,
    bc_left=0.0,
    bc_right=0.0,
)

u_exact_ref = u_exact(x_ref)

final_error = np.linalg.norm(
    u_ref
    -
    u_exact_ref
)

logger.info(
    "Reference error = %.12e",
    final_error,
)


# ============================================================
# RESIDUAL
# ============================================================

lhs, rhs, residual = (
    solver.residual(
        x_ref,
        u_ref,
    )
)

residual_norm = np.linalg.norm(
    residual
)

logger.info(
    "Residual norm = %.12e",
    residual_norm,
)


# ============================================================
# PLOTS
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.loglog(
    mesh_sizes,
    errors,
    "o-",
    lw=2,
)

ax.grid(True)

ax.set_xlabel(
    "Number of elements"
)

ax.set_ylabel(
    "RMS error"
)

ax.set_title(
    "FEM1D Convergence"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "convergence_validation.png",
    dpi=600,
)

plt.close()


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    x_ref,
    u_exact_ref,
    lw=3,
    label="Exact",
)

ax.plot(
    x_ref,
    u_ref,
    "--",
    lw=2,
    label="FEM",
)

ax.grid(True)
ax.legend()

plt.tight_layout()

plt.savefig(
    run_dir /
    "solution_validation.png",
    dpi=600,
)

plt.close()


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    x_ref,
    residual,
    lw=2,
)

ax.grid(True)

ax.set_xlabel("x")
ax.set_ylabel("Residual")

plt.tight_layout()

plt.savefig(
    run_dir /
    "residual_validation.png",
    dpi=600,
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

logger.info(
    "Validation completed"
)

logger.info(
    "Average order = %.6f",
    average_order,
)

logger.info(
    "Residual norm = %.6e",
    residual_norm,
)

print()
print("=" * 70)
print("FEM1D VALIDATION PASSED")
print("=" * 70)
print(f"Average order : {average_order:.4f}")
print(f"Residual norm : {residual_norm:.4e}")
print(f"Run directory : {run_dir}")
print("=" * 70)