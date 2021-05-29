"""
Tests to verify that the Polyline class is well-behaved.
"""

import logging

import numpy as np
import pytest

from zu.polyline import Polyline


logging.getLogger().setLevel(logging.DEBUG)


# radius


def test_no_points_given() -> None:
    """Tests that if there are no control points in the polyline, then a
    ValueError should be raised.
    """
    with pytest.raises(ValueError):
        Polyline(np.array([]))


def test_single_point_polyline_radius() -> None:
    """Tests that along the entire line, a polyline defined by a single
    point should be equal to just that single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        assert (curve.radius_at(parameter) == control_points[0]).all(), (
            f"Fails to say that a polyline defined by {control_points} is "
            f"equal to {control_points[0]} at parameter {parameter}."
        )


def test_single_point_non_zero_polyline_radius() -> None:
    """Tests that along the entire line, a polyline defined by a single
    point should be equal to just that single point where that point is
    not the origin.
    """
    control_points = np.array([(1.0, -2.0, -4.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        assert (curve.radius_at(parameter) == control_points[0]).all(), (
            f"Fails to say that a polyline defined by {control_points} is "
            f"equal to {control_points[0]} at parameter {parameter}."
        )


def test_single_axis_polyline_radius() -> None:
    """Tests that a polyline that only varies along a single axis works."""
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_two_axis_polyline_radius() -> None:
    """Tests that a polyline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter * (2.0 ** -0.5), parameter * (2.0 ** -0.5), 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_three_axis_polyline_radius() -> None:
    """Tests that a simple polyline that varies in a 3D coordinate
    system works.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_multi_segment_polyline_radius() -> None:
    """Tests that a polyline made from multiple segments works."""
    # a length=1.0 curve with segments along each axis.  Each segment is
    # length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 3.0, num=10):
        position = (
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * parameter)),
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * (parameter - 1.0))),
            min(1.0 / 3.0, max(0.0, 1.0 / 3.0 * (parameter - 2.0))),
        )
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg="Fails to find the position along a multi-segment "
            "polyline.",
        )


def test_multi_segment_non_axis_aligned_polyline_radius() -> None:
    """Tests that a multi-segment polyline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment polyline that isn't axis aligned.  Each
    # segment is length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = Polyline(control_points)
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

        position = control_points[initial_vertex] * (1 - local_parameter) + (
            control_points[final_vertex] * local_parameter
        )
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg="Fails to find the position along a multi-segment "
            "non-axis-aligned polyline.",
        )


def test_non_unit_length_polyline_radius() -> None:
    """Tests that a non-unit length polyline has its parameter go up to
    the number of segments of the polyline.
    """
    # a multi-segment polyline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 3.0, num=13):
        position = (
            min(2.0, max(0.0, 2.0 * parameter)),
            min(1.0, max(0.0, parameter - 1.0)),
            min(1.0, max(0.0, parameter - 2.0)),
        )
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg="Fails to find the position along a multi-segment "
            "polyline with non-unit length.",
        )


def test_repeated_vertex_polyline_radius() -> None:
    """Tests that a polyline with a repeated vertex will have its
    position at that vertex for 1.0 parameter distance.
    """
    # a polyline with a repeated vertex and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 4.0, num=17):
        position = (
            min(2.0, max(0.0, 2.0 * (parameter - 1.0))),
            min(1.0, max(0.0, parameter - 2.0)),
            min(1.0, max(0.0, parameter - 3.0)),
        )
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg="Fails to find the position along a polyline with a "
            "repeated vertex.",
        )


# first derivative


def test_single_point_polyline_first_derivative() -> None:
    """Tests that along the entire line, a first derivative of a
    polyline defined by a single point should be equal to just that
    single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a polyline defined by "
                f"{control_points} has first derivative equal to "
                f"{control_points[0]} at parameter {parameter}."
            ),
        )


def test_single_point_non_zero_polyline_first_derivative() -> None:
    """Tests that along the entire line, a polyline defined by a single
    point should should have first derivative equal to zero.
    """
    control_points = np.array([(1.0, -2.0, -4.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [0, 0, 0],
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to [0, 0, 0] at parameter "
            f"{parameter}.",
        )


def test_single_axis_polyline_first_derivative() -> None:
    """Tests that a polyline that only varies along a single axis has
    constant first derivative.
    """
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [1, 0, 0],
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_two_axis_polyline_first_derivative() -> None:
    """Tests that a polyline that varies on a single planar axis has
    constant first derivative.
    """
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [2.0 ** -0.5, 2.0 ** -0.5, 0.0],
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to {[2.0 ** -0.5, 2.0 ** -0.5, 0.0]} "
            f"at parameter {parameter}.",
        )


def test_three_axis_polyline_first_derivative() -> None:
    """Tests that a polyline that varies on a single spacial axis has
    constant first derivative.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5],
            err_msg=f"Fails to say that a polyline defined by "
            f"{control_points} is equal to "
            "{[3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5]} at parameter "
            f"{parameter}.",
        )


def test_multi_segment_polyline_first_derivative() -> None:
    """Tests that a polyline made from multiple segments has correct
    first derivative.
    """
    # a length=1.0 curve with segments along each axis.  Each segment is
    # length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 3.0, num=10):
        velocity = (
            1 / 3 if parameter < 1 else 0,
            1 / 3 if 1 <= parameter < 2 else 0,
            1 / 3 if parameter >= 2 else 0,
        )
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            velocity,
            err_msg="Fails to find the velocity along a multi-segment "
            "polyline.",
        )


def test_multi_segment_non_axis_aligned_polyline_first_derivative() -> None:
    """Tests that a multi-segment polyline with non-axis-aligned
    segments has correct first derivative.
    """
    # a length=1.0 3-segment polyline that isn't axis aligned.  Each
    # segment is length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 3.0, num=13):
        if parameter < 1.0:
            initial_vertex = 0
            final_vertex = 1
        elif parameter < 2.0:
            initial_vertex = 1
            final_vertex = 2
        elif parameter <= 3.0:
            initial_vertex = 2
            final_vertex = 3

        velocity = (
            control_points[final_vertex] - control_points[initial_vertex]
        )
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            velocity,
            err_msg="Fails to find the velocity along a multi-segment "
            "non-axis-aligned polyline.",
        )


def test_non_unit_length_polyline_first_derivative() -> None:
    """Tests that a non-unit length polyline has its parameter go up to
    the number of segments of the polyline and gives the first
    derivative.
    """
    # a multi-segment polyline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 3.0, num=13):
        velocity = (
            2 if parameter < 1 else 0,
            1 if 1 <= parameter < 2 else 0,
            1 if parameter >= 2 else 0,
        )
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            velocity,
            err_msg="Fails to find the velocity along a multi-segment "
            "polyline with non-unit length.",
        )


def test_repeated_vertex_polyline_first_derivative() -> None:
    """Tests that a polyline with a repeated vertex will have its
    first derivative be zero at that vertex for 1.0 parameter distance.
    """
    # a polyline with a repeated vertex and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 4.0, num=17):
        velocity = (
            2 if 1 <= parameter < 2 else 0,
            1 if 2 <= parameter < 3 else 0,
            1 if parameter >= 3 else 0,
        )
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            velocity,
            err_msg="Fails to find the velocity along a polyline with a "
            f"repeated vertex at parameter {parameter}.",
        )


# higher-order derivatives


def test_second_derivative() -> None:
    """Tests that the second derivative of a polyline is always zero at
    all parameters, given the polyline has at least 1 vertex.
    """
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 4.0, num=17):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [0, 0, 0],
            err_msg="Fails to give the second derivative of a polyline "
            "as being [0, 0, 0].",
        )


def test_third_derivative() -> None:
    """Tests that the third derivative of a polyline is always zero at
    all parameters, given the polyline has at least 1 vertex.
    """
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    for parameter in np.linspace(0.0, 4.0, num=17):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [0, 0, 0],
            err_msg="Fails to give the third derivative of a polyline "
            "as being [0, 0, 0].",
        )


def test_below_bounds_raises() -> None:
    """If a parameter that's passed in has value < 0.0, then it should
    raise Polyline.BelowBounds.
    """
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    with pytest.raises(Polyline.BelowBounds):
        curve.radius_at(-0.0001)


def test_above_bounds_raises() -> None:
    """If a parameter that's passed in has value greater than the index
    of the last control point, then it should raise
    Polyline.AboveBounds.
    """
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = Polyline(control_points)
    with pytest.raises(Polyline.AboveBounds):
        curve.radius_at(3.0001)
