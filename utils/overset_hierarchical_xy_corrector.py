#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np


@dataclass
class HierarchicalCorrectionResult:
    correction_x: np.ndarray
    correction_y: np.ndarray
    correction_xy: np.ndarray
    correction_total: np.ndarray
    corrected: np.ndarray


@dataclass
class DomainCorrectionData:
    name: str
    u: np.ndarray
    correction_x: np.ndarray = None
    correction_y: np.ndarray = None


class OversetHierarchicalXYCorrector:
    """
    Generic hierarchical corrector.

    e = ex + ey + exy

    exy = ex * ey

    No additional unknowns are introduced.
    The tensor-product contribution is generated from the
    product of x- and y-direction homogeneous solutions.
    """

    @staticmethod
    def combine(u, correction_x=None, correction_y=None):

        if correction_x is None:
            correction_x = np.zeros_like(u)

        if correction_y is None:
            correction_y = np.zeros_like(u)

        correction_xy = correction_x * correction_y

        correction_total = (
            correction_x
            + correction_y
            + correction_xy
        )

        corrected = u - correction_total

        return HierarchicalCorrectionResult(
            correction_x=correction_x,
            correction_y=correction_y,
            correction_xy=correction_xy,
            correction_total=correction_total,
            corrected=corrected,
        )


class FourDomainLCorrector:
    """

         D3 | D4
        ---------
         D1 | D2

    Each domain receives:

    ex : vertical-interface correction
    ey : horizontal-interface correction

    and the corner contribution

    exy = ex*ey

    automatically.
    """

    @staticmethod
    def apply(
        u1,
        u2,
        u3,
        u4,
        res12,
        res34,
        res13,
        res24,
    ):

        corr_x_D1 = u1 - res12.corrected_left
        corr_x_D2 = u2 - res12.corrected_right
        corr_x_D3 = u3 - res34.corrected_left
        corr_x_D4 = u4 - res34.corrected_right

        corr_y_D1 = res12.corrected_left - res13.corrected_left
        corr_y_D2 = res12.corrected_right - res24.corrected_left
        corr_y_D3 = res34.corrected_left - res13.corrected_right
        corr_y_D4 = res34.corrected_right - res24.corrected_right

        D1 = OversetHierarchicalXYCorrector.combine(
            u1,
            corr_x_D1,
            corr_y_D1,
        )

        D2 = OversetHierarchicalXYCorrector.combine(
            u2,
            corr_x_D2,
            corr_y_D2,
        )

        D3 = OversetHierarchicalXYCorrector.combine(
            u3,
            corr_x_D3,
            corr_y_D3,
        )

        D4 = OversetHierarchicalXYCorrector.combine(
            u4,
            corr_x_D4,
            corr_y_D4,
        )

        return {
            'D1': D1,
            'D2': D2,
            'D3': D3,
            'D4': D4,
        }


# usage:
#
# res12 = OversetInterfaceCorrector2D.correct(... direction='x')
# res34 = OversetInterfaceCorrector2D.correct(... direction='x')
# res13 = OversetInterfaceCorrector2D.correct(... direction='y')
# res24 = OversetInterfaceCorrector2D.correct(... direction='y')
#
# hierarchy = FourDomainLCorrector.apply(
#     u1,u2,u3,u4,
#     res12,res34,res13,res24,
# )
#
# u1f = hierarchy['D1'].corrected
# u2f = hierarchy['D2'].corrected
# u3f = hierarchy['D3'].corrected
# u4f = hierarchy['D4'].corrected
