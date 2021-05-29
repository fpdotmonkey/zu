"""
Bits for creating spline curves.

Class BSplineCurve:
    A class that creates, maniuplates, and queries basis-spline curves.
"""

import logging
from typing import Callable, Optional

import numpy as np

from zu.analytic_curve import AnalyticCurve


class BSplineCurve(AnalyticCurve):
    """B-Spline curves are geometric curves defined on a set of points
    that they pass through, and typically its traversal through those
    points is smooth.

    Sometimes they're referred to as NURBS curves (Non-Uniform Rational
    B-Spline)

    They're commonly used in vector graphics and CAD programs to
    represent arbitrary curves and fonts.

    B-Spline curves are constructed from basis spline functions, which
    are scalar, typically smooth functions used for interpolation.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        knot_vector: Optional[np.ndarray] = None,
        order: int = 3,
        periodic: bool = False,
        cyclic_closed: bool = False,
    ) -> None:
        """Constructs a new B-Spline curve

        :param      control_points: The control_points defining the spline.
        :type       control_points: numpy.ndarray
        :param      order:          The order of the spline.
        :type       order:          int
        :param      knot_vector:    The knot the spline is constructed on.
        :type       knot_vector:    numpy.ndarray
        """
