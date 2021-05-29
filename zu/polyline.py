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
        self._set_control_point_parameters_to(
            np.array([float(i) for i in range(self._number_of_control_points)])
        )

        super().__init__(
            (
                self._radius_function(self._control_points),
                self._first_derivative_function(self._control_points),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(0, self._number_of_control_points + 1),
        )

    def _radius_function(
        self, control_points: np.ndarray
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
                "There is only one control point, so for all parameters, the "
                "position must be at this point %s.",
                self._control_points[0],
            )
            return lambda parameter: self._control_points[0]

        def radius(parameter: float) -> npt.ArrayLike:
            """
            Computes the radius of a general polyline.

            :param      parameter:  The parameter along the curve to
                                    take the radius vector.
            :type       parameter:  float

            :returns:   The radius vector.
            :rtype:     numpy.typing.ArrayLike
            """
            lower_control_point_index = np.where(
                self._control_point_parameters <= parameter
            )[0].max()
            if np.isclose(
                self._control_point_parameters[lower_control_point_index],
                self._control_point_parameters[-1],
            ):
                # parameter is at its max
                position = self._control_points[-1]
                logging.debug(
                    "Interpolated radius at parameter %f, which is the "
                    "end of the polyline, getting %s.",
                    parameter,
                    position,
                )
                return position
            upper_control_point_index = np.where(
                self._control_point_parameters > parameter
            )[0].min()
            lower_control_point = self._control_points[
                lower_control_point_index
            ]
            upper_control_point = self._control_points[
                upper_control_point_index
            ]
            local_parameter = (
                parameter
                - self._control_point_parameters[lower_control_point_index]
            ) / (
                self._control_point_parameters[upper_control_point_index]
                - self._control_point_parameters[lower_control_point_index]
            )

            position = lower_control_point * (1 - local_parameter) + (
                upper_control_point * local_parameter
            )
            logging.debug(
                "Interpolated radius at parameter %s, which is between "
                "%s and %s, and got %s.",
                parameter,
                lower_control_point,
                upper_control_point,
                position,
            )

            return position

        return radius

    def _first_derivative_function(
        self, control_points: np.ndarray
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
                "first derivative must be [0, 0, 0].",
            )
            return lambda parameter: np.array([0, 0, 0])

        def first_derivative(parameter: float) -> npt.ArrayLike:
            """
            Computes the first derivative of a general polyline.

            :param      parameter:  The parameter along the curve to
                                    take the radius vector.
            :type       parameter:  float

            :returns:   The first derivative vector.
            :rtype:     numpy.typing.ArrayLike
            """
            lower_control_point_index = np.where(
                self._control_point_parameters <= parameter
            )[0].max()
            if np.isclose(
                self._control_point_parameters[lower_control_point_index],
                self._control_point_parameters[-1],
            ):
                # parameter is at its max
                position = self._control_points[-1] - self._control_points[-2]
                logging.debug(
                    "Interpolated first derivative at parameter %f, "
                    "which is the end of the polyline, getting %s.",
                    parameter,
                    position,
                )
                return position
            upper_control_point_index = np.where(
                self._control_point_parameters > parameter
            )[0].min()
            lower_control_point = self._control_points[
                lower_control_point_index
            ]
            upper_control_point = self._control_points[
                upper_control_point_index
            ]

            position = upper_control_point - lower_control_point
            logging.debug(
                "Calculating the first derivative as the difference "
                "between the position of the two adjacent control_points.",
            )

            return position

        return first_derivative

    def _set_control_point_parameters_to(self, parameters: np.ndarray) -> None:
        """Sets the parameters that each control point is at.  The input list
        of parameters must be the same length as the the number of
        control_points or else this will throw an AssertionError.

        :param      parameters:     The parameters that each control point is
                                    located at.
        :type       parameters:     numpy.ndarray

        :raises     AssertionError: If the number of parameter is not
                                    the same as the number of control_points.
        """
        assert len(parameters) == self._number_of_control_points, (
            "There are not enought parameter lengths for the number of "
            "control_points."
        )
        logging.debug("Setting control point parameters to %s", parameters)
        self._control_point_parameters = parameters
