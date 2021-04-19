"""
Tests to verify that the Polyline class is well-behaved.
"""

import logging
from random import random

import numpy as np

from zu.polyline import Polyline


logging.getLogger().setLevel(logging.DEBUG)


def test_no_points_given():
    """Tests that if there are no vertices in the polyline, then its
    position at any parameter should be NaN.
    """
    vertices = np.array([])
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        assert np.isnan(curve.radius_at(parameter)).all(), (
            "Fails to say that any position on a polyline with no "
            "vertices is NaN."
        )


def test_single_point_polyline():
    """Tests that along the entire line, a polyline defined by a single
    point should be equal to just that single point.
    """
    vertices = np.array([(0.0, 0.0, 0.0)])
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        assert (curve.radius_at(parameter) == vertices[0]).all(), (
            f"Fails to say that a polyline defined by {vertices} is "
            f"equal to {vertices[0]} at parameter {parameter}."
        )


def test_single_point_non_zero_polyline():
    """Tests that along the entire line, a polyline defined by a single
    point should be equal to just that single point where that point is
    not the origin.
    """
    vertices = np.array([(1.0, -2.0, -4.0)])
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        assert (curve.radius_at(parameter) == vertices[0]).all(), (
            f"Fails to say that a polyline defined by {vertices} is "
            f"equal to {vertices[0]} at parameter {parameter}."
        )


def test_single_axis_polyline():
    """Tests that a polyline that only varies along a single axis works."""
    vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter, 0.0, 0.0)
        assert (curve.radius_at(parameter) == position).all(), (
            f"Fails to say that a polyline defined by {vertices} is "
            f"equal to {position} at parameter {parameter}."
        )


def test_two_axis_polyline():
    """Tests that a polyline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    vertices = np.array([(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)])
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter * (2.0 ** -0.5), parameter * (2.0 ** -0.5), 0.0)
        assert (curve.radius_at(parameter) == position).all(), (
            f"Fails to say that a polyline defined by {vertices} is "
            f"equal to {position} at parameter {parameter}."
        )


def test_three_axis_polyline():
    """Tests that a simple polyline that varies in a 3D coordinate
    system works.
    """
    # a length=1.0 curve in 3D
    vertices = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        assert (curve.radius_at(parameter) == position).all(), (
            f"Fails to say that a polyline defined by {vertices} is "
            f"equal to {position} at parameter {parameter}."
        )


def test_multi_segment_polyline():
    """Tests that a polyline made from multiple segments works."""
    # a length=1.0 curve with segments along each axis.  Each segment is
    # length=1/3.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 3.0, num=10):
        position = (
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * parameter)),
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * (parameter - 1.0))),
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * (parameter - 2.0))),
        )
        assert (
            curve.radius_at(parameter) == position
        ).all(), "Fails to find the position along a multi-segment polyline."


def test_multi_segment_non_axis_aligned_polyline():
    """Tests that a multi-segment polyline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment polyline that isn't axis aligned.  Each
    # segment is length=1/3.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 3.0, num=13):
        if parameter < 1.0:
            initial_vertex = 0
            final_vertex = 1
            local_parameter = parameter
        elif parameter < 2.0:
            initial_vertex = 1
            final_vertex = 2
            local_parameter = parameter - 1.0
        elif parameter <= 3.0:
            initial_vertex = 2
            final_vertex = 3
            local_parameter = parameter - 2.0
        else:
            raise ValueError("Parameter outside of [0, 3]")

        position = vertices[initial_vertex] * (1 - local_parameter) + (
            vertices[final_vertex] * local_parameter
        )
        assert (curve.radius_at(parameter) == position).all(), (
            "Fails to find the position along a multi-segment "
            "non-axis-aligned polyline."
        )


def test_non_unit_length_polyline():
    """Tests that a non-unit length polyline has its parameter go up to
    the number of segments of the polyline.
    """
    # a multi-segment polyline with length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 3.0, num=13):
        position = (
            min(2.0, max(0.0, 2.0 * parameter)),
            min(1.0, max(0.0, parameter - 1.0)),
            min(1.0, max(0.0, parameter - 2.0)),
        )
        assert (curve.radius_at(parameter) == position).all(), (
            "Fails to find the position along a multi-segment polyline "
            "with non-unit length."
        )


def test_repeated_vertex_polyline():
    """Tests that a polyline with a repeated vertex will have its
    position at that vertex for 1.0 parameter distance.
    """
    # a polyline with a repeated vertex and length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(vertices)
    for parameter in np.linspace(0.0, 4.0, num=17):
        position = (
            min(2.0, max(0.0, 2.0 * (parameter - 1.0))),
            min(1.0, max(0.0, parameter - 2.0)),
            min(1.0, max(0.0, parameter - 3.0)),
        )
        assert (curve.radius_at(parameter) == position).all(), (
            "Fails to find the position along a polyline with a "
            "repeated vertex."
        )


def test_below_bounds_parameters_give_endpoints():
    """Tests that if a parameter is given that is below the nominal
    range of the parameter, then it will return the first endpoint.
    """
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(vertices)
    assert (curve.radius_at(-1.0) == (0.0, 0.0, 0.0)).all(), (
        "Fails to return the position of the first vertex when a "
        "below-bounds parameter is given."
    )


def test_above_bounds_parameters_give_endpoints():
    """Tests that if a parameter is given that is above the nominal
    range of the parameter, then it will return the last endpoint.
    """
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(vertices)
    assert (curve.radius_at(10.0) == (2.0, 1.0, 1.0)).all(), (
        "Fails to return the position of the last vertex when a "
        "above-bounds parameter is given."
    )
