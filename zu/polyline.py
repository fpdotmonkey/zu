"""
A polyline class.
"""

from __future__ import annotations
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from zu.analytic_curve import AnalyticCurve


class Polyline(AnalyticCurve):
    """A curve defined by an ordered series of vertices and the
    straight-line paths between them.
    """

    def __init__(self, vertices: npt.ArrayLike) -> None:
        """Creates the polyline with a parameterization of 1 unit per
        edge.
        """
        if len(vertices) == 0:
            raise ValueError(
                "There must be at least one vertex on a polyline."
            )
        self._vertices: npt.ArrayLike = vertices
        self._number_of_vertices: int = len(self._vertices)
        self._set_vertex_parameters_to(
            np.array([float(i) for i in range(self._number_of_vertices)])
        )

        super().__init__(
            self,
            (
                self._radius_function(self._vertices),
                self._first_derivative_function(self._vertices),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(0, self._number_of_vertices + 1),
        )

    def _radius_function(
        self, vertices: npt.ArrayLike
    ) -> Callable[[float], npt.ArrayLike]:
        """Computes a polyline that goes through each of the vertices
        and returns a function that gives the line's coordinates given a
        parameter.
        """
        if len(vertices) == 1:
            logging.debug(
                "There is only one vertex, so for all parameters, the "
                "position must be at this point %s.",
                self._vertices[0],
            )
            return lambda parameter: self._vertices[0]

        def radius(parameter: float) -> npt.ArrayLike:
            """Computes the radius of a general polyline."""
            lower_vertex_index = np.where(
                self._vertex_parameters <= parameter
            )[0].max()
            if np.isclose(
                self._vertex_parameters[lower_vertex_index],
                self._vertex_parameters[-1],
            ):
                # parameter is at its max
                position = self._vertices[-1]
                logging.debug(
                    "Interpolated at parameter %f, which is the end of the "
                    "polyline, getting %s.",
                    parameter,
                    position,
                )
                return position
            upper_vertex_index = np.where(self._vertex_parameters > parameter)[
                0
            ].min()
            lower_vertex = self._vertices[lower_vertex_index]
            upper_vertex = self._vertices[upper_vertex_index]
            local_parameter = (
                parameter - self._vertex_parameters[lower_vertex_index]
            ) / (
                self._vertex_parameters[upper_vertex_index]
                - self._vertex_parameters[lower_vertex_index]
            )

            position = lower_vertex * (1 - local_parameter) + (
                upper_vertex * local_parameter
            )
            logging.debug(
                "Interpolated at parameter %s, which is between %s and %s, "
                "and got %s.",
                parameter,
                lower_vertex,
                upper_vertex,
                position,
            )

            return position

    def _first_derivative_function(
        self, vertices: npt.ArrayLike
    ) -> Callable[[float], npt.ArrayLike]:
        """Computes a function that gives the rate of change of the
        curve with respect to the parameter and return it.
        """
        if len(vertices) == 1:
            logging.debug(
                "There is only one vertex, so for all parameters, the "
                "position must be at this point %s.",
                self._vertices[0],
            )
            return lambda parameter: np.array([0, 0, 0])

    # def radius_at(self, parameter: float) -> npt.ArrayLike:
    #     """Gives the position of the point at the parameter."""
    #     if self._number_of_vertices < 1:
    #         logging.debug(
    #             "There are no vertices, the position defaults to NaN."
    #         )
    #         return np.nan
    #     if self._number_of_vertices == 1:
    #         logging.debug(
    #             "There is only one vertex, so for all parameters, the "
    #             "position must be at this point %s.",
    #             self._vertices[0],
    #         )
    #         return self._vertices[0]
    #     if parameter < 0.0:
    #         logging.debug(
    #             "Parameter %f is below the minimum parameter 0.0.  "
    #             "Setting position to the first vertex %s.",
    #             parameter,
    #             self._vertices[0],
    #         )
    #         return self._vertices[0]
    #     if parameter > self._number_of_edges:
    #         logging.debug(
    #             "Parameter %f is above the max parameter %d.  Setting "
    #             "position to the last vertex %s.",
    #             parameter,
    #             self._number_of_edges,
    #             self._vertices[-1],
    #         )
    #         return self._vertices[-1]

    #     lower_vertex_index = np.where(self._vertex_parameters <= parameter)[
    #         0
    #     ].max()
    #     if np.isclose(
    #         self._vertex_parameters[lower_vertex_index],
    #         self._vertex_parameters[-1],
    #     ):
    #         # parameter is at its max
    #         position = self._vertices[-1]
    #         logging.debug(
    #             "Interpolated at parameter %f, which is the end of the "
    #             "polyline, getting %s.",
    #             parameter,
    #             position,
    #         )
    #     else:
    #         upper_vertex_index = np.where(self._vertex_parameters > parameter)[
    #             0
    #         ].min()
    #         lower_vertex = self._vertices[lower_vertex_index]
    #         upper_vertex = self._vertices[upper_vertex_index]
    #         local_parameter = (
    #             parameter - self._vertex_parameters[lower_vertex_index]
    #         ) / (
    #             self._vertex_parameters[upper_vertex_index]
    #             - self._vertex_parameters[lower_vertex_index]
    #         )

    #         position = lower_vertex * (1 - local_parameter) + (
    #             upper_vertex * local_parameter
    #         )
    #         logging.debug(
    #             "Interpolated at parameter %s, which is between %s and %s, "
    #             "and got %s.",
    #             parameter,
    #             lower_vertex,
    #             upper_vertex,
    #             position,
    #         )

    #     return position

    def _set_vertex_parameters_to(self, parameters: npt.ArrayLike) -> None:
        """Sets the parameters that each vertex is at.  The input list
        of parameters must be the same length as the the number of
        vertices or else this will throw an AssertionError.
        """
        assert len(parameters) == self._number_of_vertices, (
            "There are not enought parameter lengths for the number of "
            "vertices."
        )
        logging.debug("Setting vertex parameters to %s", parameters)
        self._vertex_parameters = parameters
