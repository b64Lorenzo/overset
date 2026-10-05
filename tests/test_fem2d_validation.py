"""
test_fem2d_validation.py
========================

Validation script for

    src.problems.elliptic2d
    src.solver.fem2d

Author
------
Lorenzo Zambelli
"""

import numpy as np
import matplotlib.pyplot as plt

from skfem import BilinearForm
from skfem import LinearForm
from skfem.helpers import dot, grad

from src.utils.logger import Logger

from src.solver.fem2d import (
    solve_rect,
)

from src.problems.elliptic2d import (
    Elliptic2DProblem,
    Elliptic2DCoefficients,
)


# ============================================================
# LOGGER
# ============================================================

log_mgr = Logger(
    name="test_fem2d_validation",
    run_prefix="test_fem2d_validation",
)

logger = log_mgr.get_logger()
run_dir = log_mgr.get_run_dir()


# ============================================================
# MANUFACTURED PROBLEM
# ============================================================

k_reaction = 25.0


class ManufacturedProblem(Elliptic2DProblem):

    def h(self, x, y):
        return 1.0

    def hx(self, x, y):
        return 0.0

    def hy(self, x, y):
        return 0.0

    def p(self, x, y):
        return 1.0

    def px(self, x, y):
        return 0.0

    def py(self, x, y):
        return 0.0

    def fxx(self, x, y, t=0.0):
        return 0.5 * rhs_exact(x, y)

    def fyy(self, x, y, t=0.0):
        return 0.5 * rhs_exact(x, y)

    def fx(self, x, y, t=0.0):
        return 0.0

    def fy(self, x, y, t=0.0):
        return 0.0


# ============================================================
# EXACT SOLUTION
# ============================================================

def u_exact(x, y):

    return (
        np.sin(np.pi * x)
        * np.sin(np.pi * y)
    )


def rhs_exact(x, y):

    return (
        k_reaction
        + 2.0 * np.pi**2
    ) * u_exact(x, y)


problem = ManufacturedProblem(
    Elliptic2DCoefficients(
        k=k_reaction
    )
)


# ============================================================
# FE FORMS
# ============================================================

@BilinearForm
def diffusion(u, v, w):

    return dot(
        grad(u),
        grad(v),
    )


@BilinearForm
def reaction(u, v, w):

    return k_reaction * u * v


@LinearForm
def rhs(v, w):

    return (
        rhs_exact(
            w.x[0],
            w.x[1],
        )
        * v
    )


# ============================================================
# PROBLEM DIAGNOSTICS
# ============================================================

logger.info(
    "Problem summary: %s",
    problem.summary(),
)

x0 = 0.5
y0 = 0.5

lam = problem.decay_parameter(
    x0,
    y0,
)

delta = problem.boundary_layer_thickness(
    x0,
    y0,
)

logger.info(
    "Decay parameter lambda = %.6f",
    lam,
)

logger.info(
    "Boundary layer thickness delta = %.6f",
    delta,
)

assert np.isclose(
    lam,
    np.sqrt(k_reaction),
)

assert np.isclose(
    delta,
    1/np.sqrt(k_reaction),
)


# ============================================================
# CONVERGENCE STUDY
# ============================================================

mesh_sizes = [
    8,
    16,
    32,
    64,
]

errors = []

for n in mesh_sizes:

    mesh, basis, u = solve_rect(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        nx=n,
        ny=n,
        diffusion_form=diffusion,
        reaction_form=reaction,
        rhs_form=rhs,
        boundary_conditions={
            "W": lambda x, y: 0.0,
            "E": lambda x, y: 0.0,
            "S": lambda x, y: 0.0,
            "N": lambda x, y: 0.0,
        },
    )

    X = basis.doflocs[0]
    Y = basis.doflocs[1]

    u_ref = u_exact(
        X,
        Y,
    )

    error = np.sqrt(
        np.mean(
            (u - u_ref) ** 2
        )
    )

    errors.append(error)

    logger.info(
        "N=%4d error=%.12e",
        n,
        error,
    )


# ============================================================
# OBSERVED ORDER
# ============================================================

orders = []

for k in range(
    1,
    len(errors),
):

    p = (
        np.log(
            errors[k-1]
            /
            errors[k]
        )
        /
        np.log(2.0)
    )

    orders.append(p)

    logger.info(
        "order[%d]=%.6f",
        k,
        p,
    )

average_order = np.mean(
    orders
)

logger.info(
    "Average order = %.6f",
    average_order,
)

assert average_order > 1.6


# ============================================================
# HIGH RESOLUTION SOLUTION
# ============================================================

mesh, basis, u = solve_rect(
    xmin=0.0,
    xmax=1.0,
    ymin=0.0,
    ymax=1.0,
    nx=100,
    ny=100,
    diffusion_form=diffusion,
    reaction_form=reaction,
    rhs_form=rhs,
    boundary_conditions={
        "W": lambda x, y: 0.0,
        "E": lambda x, y: 0.0,
        "S": lambda x, y: 0.0,
        "N": lambda x, y: 0.0,
    },
)

X = basis.doflocs[0]
Y = basis.doflocs[1]

u_ref = u_exact(X, Y)

final_error = np.linalg.norm(
    u - u_ref
)

logger.info(
    "Reference error = %.12e",
    final_error,
)


# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(8, 6))

plt.loglog(
    mesh_sizes,
    errors,
    "o-",
    lw=3,
)

plt.xlabel(
    "Elements per direction"
)

plt.ylabel(
    "RMS error"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    run_dir /
    "convergence_validation.png",
    dpi=600,
)

plt.close()


fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

ax.plot_trisurf(
    X,
    Y,
    u,
    cmap="viridis",
)

ax.set_title(
    "FEM Solution"
)

plt.tight_layout()

plt.savefig(
    run_dir /
    "solution_surface.png",
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
    "Reference error = %.12e",
    final_error,
)

print()
print("=" * 70)
print("FEM2D VALIDATION PASSED")
print("=" * 70)
print(f"Average order : {average_order:.4f}")
print(f"Reference error : {final_error:.4e}")
print(f"Run directory : {run_dir}")
print("=" * 70)