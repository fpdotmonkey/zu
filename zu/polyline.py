"""
Stuff for computing curves that are straight lines between points.

Class Polyline:
    A class the represents such a sequence of straight lines between
    points.
"""

from __future__ import annotations
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from zu.analytic_curve import AnalyticCurve


class Polyline(AnalyticCurve):
    """A curve defined by an ordered series of control_points and the
    straight-line paths between them.
    """

    def __init__(self, control_points: np.ndarray) -> None:
        """Creates the polyline with a parameterization of 1 unit per
        edge.

        :param      control_points:  The points through which the polyline
                                     will go.
        :type       control_points:  numpy.ndarray
        """
        if control_points.shape[0] == 0:
            raise ValueError(
                "There must be at least one control point on a polyline."
            )

        self._control_points: np.ndarray = control_points
        self._number_of_control_points = self._control_points.shape[0]
        print(self._number_of_control_points - 1)
        super().__init__(
            (
                _radius_function(self._control_points),
                _first_derivative_function(self._control_points - 1),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(0, self._number_of_control_points),
        )


def _radius_function(
    control_points: np.ndarray,
) -> Callable[[float], npt.ArrayLike]:
    """Computes a polyline that goes through each of the control_points
    and returns a function that gives the line's coordinates given a
    parameter.

    :param      control_points:  The points through which the polyline
                                 will go.
    :type       control_points:  numpy.ndarray

    :returns:   A function that takes a parameter and returns a
                radius vector.
    :rtype:     Callable[[float], npt.ArrayLike]
    """
    if control_points.shape[0] == 1:
        logging.debug(
            "There is only one control point, so for all "
            "parameters, the radius of the polyline at this point "
            "is %s.",
            control_points[0],
        )
        return lambda parameter: control_points[0]

    def radius(parameter: float) -> npt.ArrayLike:
        """Computes the radius of a general polyline.

        :param      parameter:  The parameter along the curve to
                                take the radius vector.
        :type       parameter:  float

        :returns:   The radius vector.
        :rtype:     numpy.typing.ArrayLike
        """
        index, local_parameter = np.divmod(parameter, 1)
        index = index.astype(int)
        if index == control_points.shape[0] - 1:
            logging.debug(
                "Interpolated radius at parameter %f, which is the "
                "end of the polyline, getting %s.",
                parameter,
                control_points[-1],
            )
            return control_points[-1]
        radius = control_points[index] * (1 - local_parameter) + (
            control_points[index + 1] * local_parameter
        )
        logging.debug(
            "Computing in the general case, parameter %f gives "
            "the polyline radius of %s.",
            parameter,
            radius,
        )
        return radius

    return radius


def _first_derivative_function(
    control_points: np.ndarray,
) -> Callable[[float], npt.ArrayLike]:
    """Computes a function that gives the rate of change of the
    curve with respect to the parameter and return it.

    :param      control_points:  The points through which the polyline
                                 will go.
    :type       control_points:  numpy.ndarray

    :returns:   A vector in the direction of the first derivative.
    :rtype:     numpy.typing.ArrayLike
    """
    if control_points.shape[0] == 1:
        logging.debug(
            "There is only one control point, so for all parameters, the "
            "first derivative of the polyline must be [0, 0, 0].",
        )
        return lambda parameter: np.array([0, 0, 0])

    def first_derivative(parameter: float) -> npt.ArrayLike:
        """
        Computes the first derivative of a general polyline.

        :param      parameter:  The parameter along the curve to
                                take the first derivative vector.
        :type       parameter:  float

        :returns:   The first derivative vector.
        :rtype:     numpy.typing.ArrayLike
        """
        index = np.floor(parameter)
        index = index.astype(int)
        if index == control_points.shape[0] - 1:
            first_derivative = control_points[-1] - control_points[-2]
            logging.debug(
                "Interpolated first derivative at parameter %f, "
                "which is the end of the polyline, getting %s.",
                parameter,
                first_derivative,
            )
            return first_derivative
        first_derivative = control_points[index + 1] - control_points[index]
        logging.debug(
            "Computing in the general case, parameter %f gives "
            "the polyline first derivative of %s.",
            parameter,
            first_derivative,
        )
        return first_derivative

    return first_derivative
