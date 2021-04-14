"""
A polyline class.
"""

from __future__ import annotations
import logging

import numpy as np
import numpy.typing as npt


class Polyline:
    """A curve defined by an ordered series of vertices and the
    straight-line paths between them.
    """

    def __init__(
        self, vertices: npt.ArrayLike, natural_parameterization=False
    ) -> None:
        """Creates the polyline with a parameterization of 1 unit per
        edge.
        """
        self._vertices = vertices
        self._number_of_vertices = len(self._vertices)
        self._number_of_edges = self._number_of_vertices - 1
        self._natural_parameterization = natural_parameterization
        self._set_vertex_parameters(
            np.array([float(i) for i in range(self._number_of_vertices)])
            if not natural_parameterization
            else np.add.accumulate(self._lengths_of_each_edge())
        )

    def position_at(self, parameter: float) -> npt.ArrayLike:
        """Gives the position of the point at the parameter."""
        if self._number_of_vertices < 1:
            logging.debug(
                "There are no vertices, the position defaults to NaN."
            )
            return np.nan
        if self._number_of_vertices == 1:
            logging.debug(
                "There is only one vertex, so for all parameters, the "
                "position must be at this point %s.",
                self._vertices[0],
            )
            return self._vertices[0]
        if parameter < 0.0:
            logging.debug(
                "Parameter %f is below the minimum parameter 0.0.  "
                "Setting position to the first vertex %s.",
                parameter,
                self._vertices[0],
            )
            return self._vertices[0]
        if (
            not self._natural_parameterization
            and parameter > self._number_of_edges
        ):
            logging.debug(
                "Parameter %f is above the max parameter %d.  Setting "
                "position to the last vertex %s.",
                parameter,
                self._number_of_edges,
                self._vertices[-1],
            )
            return self._vertices[-1]
        if self._natural_parameterization and parameter > self.length:
            logging.debug(
                "Parameter %f is above the max parameter %f.  Setting "
                "position to the last vertex %s.",
                parameter,
                self.length,
                self._vertices[-1],
            )
            return self._vertices[-1]

        lower_vertex_index = np.where(self._vertex_parameters <= parameter)[
            0
        ].max()
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
        else:
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

    def naturalized(self) -> Polyline:
        """Returns a new polyline where it has natural parameterization;
        that is, the parameter goes with the arce length of the curve.
        """
        return Polyline(self._vertices, natural_parameterization=True)

    def curvature(self, parameter: float) -> float:
        """The inverse of the radius of the tangent circle at the point
        parameter.
        """
        if len(self._vertices) == 0:
            return np.nan
        del parameter  # unused, but you want to have it
        return 0.0

    @property
    def length(self) -> float:
        """The total length of the curve."""
        if self._number_of_vertices < 1:
            return np.nan
        if self._number_of_vertices == 1:
            return 0.0

        return np.sum(self._lengths_of_each_edge())

    def _set_vertex_parameters(self, parameters: npt.ArrayLike) -> None:
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

    def _lengths_of_each_edge(self) -> npt.ArrayLike:
        """Returns a list with the length of each edge of the curve."""
        return np.insert(
            np.sqrt(
                np.sum(np.square(np.diff(self._vertices, axis=0)), axis=1)
            ),
            0,
            0,
        )
