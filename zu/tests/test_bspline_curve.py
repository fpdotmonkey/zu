"""
Tests to ensure that b-spline curves are working as expected.
"""

import logging

import numpy as np
import pytest

from zu.bspline_curve import BSplineCurve

npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_bspline_no_points_given() -> None:
    """Tests that when there are no vertices in the spline, a
    ValueError is raised.
    """
    with pytest.raises(ValueError):
        BSplineCurve(np.array([]))


def test_single_point_bspline_radius() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    vertices = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(vertices, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            vertices[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {vertices} is "
                f"equal to {vertices[0]} at parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_radius() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    vertices = np.array([(1.0, -1.0, 3.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(vertices, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            vertices[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {vertices} is "
                f"equal to {vertices[0]} at parameter {parameter}."
            ),
        )


def test_single_axis_bspline_radius() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=f"Fails to say that a bspline defined by "
            f"{vertices} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_two_axis_bspline_radius() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    vertices = np.array([(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)])
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter * (2.0 ** -0.5), parameter * (2.0 ** -0.5), 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=(
                f"Fails to say that a bspline defined by {vertices} is "
                f"equal to {position} at parameter {parameter}."
            ),
        )


def test_three_axis_bspline_radius() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct radius vector.
    """
    # a length=1.0 curve in 3D
    vertices = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            position,
            err_msg=(
                f"Fails to say that a bspline defined by {vertices} is "
                f"equal to {position} at parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_radius() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct radius vector.
    """
    # a length=1.0 curve with displacements along each axis.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=10) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0992227, 0.0114312, 0.000457247],
            [0.176497, 0.0420668, 0.00365798],
            [0.234568, 0.0864198, 0.0123457],
            [0.276177, 0.139003, 0.0292638],
            [0.30407, 0.19433, 0.0571559],
            [0.320988, 0.246914, 0.0987654],
            [0.329675, 0.291267, 0.156836],
            [0.332876, 0.321902, 0.234111],
            [0.333333, 0.333333, 0.333333],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=10)):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            nominal_radii[index],
            err_msg=(
                "Fails to find the radius along a multi-displacement "
                "bspline."
            ),
        )


def test_multi_segment_non_axis_aligned_bspline_radius() -> None:
    """Tests that a multi-displacement b-spline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment b-spline that isn't axis aligned.  Each
    # segment is length=1/3.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [-0.0662307, 0.0169097, 0.035148],
            [-0.122711, 0.0424128, 0.0589616],
            [-0.16951, 0.0752816, 0.0739953],
            [-0.206699, 0.114288, 0.0828038],
            [-0.234348, 0.158206, 0.0879414],
            [-0.252528, 0.205806, 0.0919628],
            [-0.261307, 0.255861, 0.0974224],
            [-0.260757, 0.307144, 0.106875],
            [-0.250948, 0.358427, 0.122874],
            [-0.23195, 0.408483, 0.147976],
            [-0.203832, 0.456083, 0.184734],
            [-0.166667, 0.5, 0.235702],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a multi-displacement "
            "non-axis-aligned b-spline.",
        )


def test_non_unit_length_bspline_radius() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.459491, 0.0196759, 0.000578704],
            [0.842593, 0.0740741, 0.00462963],
            [1.15625, 0.15625, 0.015625],
            [1.40741, 0.259259, 0.037037],
            [1.60301, 0.376157, 0.072338],
            [1.75, 0.5, 0.125],
            [1.85532, 0.623843, 0.198495],
            [1.92593, 0.740741, 0.296296],
            [1.96875, 0.84375, 0.421875],
            [1.99074, 0.925926, 0.578704],
            [1.99884, 0.980324, 0.770255],
            [2.0, 1.0, 1.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a multi-displacement "
            "b-spline with non-unit length.",
        )


def test_repeated_vertex_bspline_radius() -> None:
    """Tests that a b-spline with a repeated vertex will have its
    position at that vertex for 1.0 parameter distance.
    """
    # a b-spline with a repeated vertex and length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.459491, 0.0196759, 0.000578704],
            [0.842593, 0.0740741, 0.00462963],
            [1.15625, 0.15625, 0.015625],
            [1.40741, 0.259259, 0.037037],
            [1.60301, 0.376157, 0.072338],
            [1.75, 0.5, 0.125],
            [1.85532, 0.623843, 0.198495],
            [1.92593, 0.740741, 0.296296],
            [1.96875, 0.84375, 0.421875],
            [1.99074, 0.925926, 0.578704],
            [1.99884, 0.980324, 0.770255],
            [2.0, 1.0, 1.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 4.0, num=17)):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a b-spline with a "
            "repeated vertex.",
        )


# first derivative


def test_single_point_bspline_first_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    vertices = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(vertices, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            vertices[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {vertices} is "
                f"equal to {vertices[0]} at parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_first_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    vertices = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(vertices, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            vertices[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {vertices} is "
                f"equal to {vertices[0]} at parameter {parameter}."
            ),
        )


def test_single_axis_bspline_first_derivative() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    vertices = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            position,
            err_msg=f"Fails to say that a bspline defined by "
            f"{vertices} is equal to {position} at parameter "
            f"{parameter}.",
        )


def test_two_axis_bspline_first_derivative() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    vertices = np.array([(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)])
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = parameter * vertices[1]
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            position,
            err_msg=(
                f"Fails to say that a bspline defined by {vertices} is "
                f"equal to {position} at parameter {parameter}."
            ),
        )


def test_three_axis_bspline_first_derivative() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct first_derivative vector.
    """
    # a length=1.0 curve in 3D
    vertices = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(vertices)
    for parameter in np.linspace(0.0, 1.0, num=5):
        position = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            position,
            err_msg=(
                f"Fails to say that a bspline defined by {vertices} is "
                f"equal to {position} at parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_first_derivative() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct first derivative vector.
    """
    # a length=1.0 curve with displacements along each axis.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=10) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0992227, 0.0114312, 0.000457247],
            [0.176497, 0.0420668, 0.00365798],
            [0.234568, 0.0864198, 0.0123457],
            [0.276177, 0.139003, 0.0292638],
            [0.30407, 0.19433, 0.0571559],
            [0.320988, 0.246914, 0.0987654],
            [0.329675, 0.291267, 0.156836],
            [0.332876, 0.321902, 0.234111],
            [0.333333, 0.333333, 0.333333],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=10)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_radii[index],
            err_msg=(
                "Fails to find the first_derivative along a "
                "multi-displacement bspline."
            ),
        )


def test_multi_segment_non_axis_aligned_bspline_first_derivative() -> None:
    """Tests that a multi-displacement b-spline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment b-spline that isn't axis aligned.  Each
    # segment is length=1/3.
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [-0.0662307, 0.0169097, 0.035148],
            [-0.122711, 0.0424128, 0.0589616],
            [-0.16951, 0.0752816, 0.0739953],
            [-0.206699, 0.114288, 0.0828038],
            [-0.234348, 0.158206, 0.0879414],
            [-0.252528, 0.205806, 0.0919628],
            [-0.261307, 0.255861, 0.0974224],
            [-0.260757, 0.307144, 0.106875],
            [-0.250948, 0.358427, 0.122874],
            [-0.23195, 0.408483, 0.147976],
            [-0.203832, 0.456083, 0.184734],
            [-0.166667, 0.5, 0.235702],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a multi-displacement "
            "non-axis-aligned b-spline.",
        )


def test_non_unit_length_bspline_first_derivative() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.459491, 0.0196759, 0.000578704],
            [0.842593, 0.0740741, 0.00462963],
            [1.15625, 0.15625, 0.015625],
            [1.40741, 0.259259, 0.037037],
            [1.60301, 0.376157, 0.072338],
            [1.75, 0.5, 0.125],
            [1.85532, 0.623843, 0.198495],
            [1.92593, 0.740741, 0.296296],
            [1.96875, 0.84375, 0.421875],
            [1.99074, 0.925926, 0.578704],
            [1.99884, 0.980324, 0.770255],
            [2.0, 1.0, 1.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a multi-displacement "
            "b-spline with non-unit length.",
        )


def test_repeated_vertex_bspline_first_derivative() -> None:
    """Tests that a b-spline with a repeated vertex will have its
    position at that vertex for 1.0 parameter distance.
    """
    # a b-spline with a repeated vertex and length=4.0
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(vertices)
    nominal_radii = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.459491, 0.0196759, 0.000578704],
            [0.842593, 0.0740741, 0.00462963],
            [1.15625, 0.15625, 0.015625],
            [1.40741, 0.259259, 0.037037],
            [1.60301, 0.376157, 0.072338],
            [1.75, 0.5, 0.125],
            [1.85532, 0.623843, 0.198495],
            [1.92593, 0.740741, 0.296296],
            [1.96875, 0.84375, 0.421875],
            [1.99074, 0.925926, 0.578704],
            [1.99884, 0.980324, 0.770255],
            [2.0, 1.0, 1.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 4.0, num=17)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_radii[index],
            err_msg="Fails to find the position along a b-spline with a "
            "repeated vertex.",
        )
