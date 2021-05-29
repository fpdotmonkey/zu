"""
Tests to ensure that b-spline curves are working as expected.
"""

# pylint: disable=too-many-lines

import logging

import numpy as np
import pytest

from zu.bspline_curve import BSplineCurve

npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_bspline_no_points_given() -> None:
    """Tests that when there are no control_points in the spline, a
    ValueError is raised.
    """
    with pytest.raises(ValueError):
        BSplineCurve(np.array([]))


def test_single_point_bspline_radius() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            control_points[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {control_points} is "
                f"equal to {control_points[0]} at parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_radius() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    control_points = np.array([(1.0, -1.0, 3.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            control_points[0],
            err_msg=(
                f"Fails to say that a b-spline defined by {control_points} is "
                f"equal to {control_points[0]} at parameter {parameter}."
            ),
        )


def test_single_axis_bspline_radius() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        radius = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            radius,
            err_msg=f"Fails to say that a bspline defined by "
            f"{control_points} is equal to {radius} at parameter "
            f"{parameter}.",
        )


def test_two_axis_bspline_radius() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        radius = (parameter * (2.0 ** -0.5), parameter * (2.0 ** -0.5), 0.0)
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            radius,
            err_msg=(
                f"Fails to say that a bspline defined by {control_points} is "
                f"equal to {radius} at parameter {parameter}."
            ),
        )


def test_three_axis_bspline_radius() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct radius vector.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        radius = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            radius,
            err_msg=(
                f"Fails to say that a bspline defined by {control_points} is "
                f"equal to {radius} at parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_radius() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct radius vector.
    """
    # a length=1.0 curve with displacements along each axis.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(control_points)
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
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(control_points)
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
            err_msg="Fails to find the radius along a multi-displacement "
            "non-axis-aligned b-spline.",
        )


def test_non_unit_length_bspline_radius() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
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
            err_msg="Fails to find the radius along a multi-displacement "
            "b-spline with non-unit length.",
        )


def test_repeated_control_point_bspline_radius() -> None:
    """Tests that a b-spline with a repeated control point will have its
    radius at that control point for 1.0 parameter distance.
    """
    # a b-spline with a repeated control point and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
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
            err_msg="Fails to find the radius along a b-spline with a "
            "repeated control point.",
        )


# first derivative


def test_single_point_bspline_first_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_first_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_axis_bspline_first_derivative() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        first_derivative = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [1.0, 0.0, 0.0],
            err_msg=f"Fails to say that a bspline defined by "
            f"{control_points} is equal to {first_derivative} at "
            f"parameter {parameter}.",
        )


def test_two_axis_bspline_first_derivative() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        first_derivative = parameter * control_points[1]
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [2.0 ** -0.5, 2.0 ** -0.5, 0.0],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {first_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_three_axis_bspline_first_derivative() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct first_derivative vector.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        first_derivative = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            [3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {first_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_first_derivative() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct first derivative vector.
    """
    # a length=1.0 curve with displacements along each axis.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_first_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=10) (inputs scaled by 1/3)
        [
            [1.0, 0.0, 0.0],
            [0.790123, 0.197531, 0.0123457],
            [0.604938, 0.345679, 0.0493827],
            [0.444444, 0.444444, 0.111111],
            [0.308642, 0.493827, 0.197531],
            [0.197531, 0.493827, 0.308642],
            [0.111111, 0.444444, 0.444444],
            [0.0493827, 0.345679, 0.604938],
            [0.0123457, 0.197531, 0.790123],
            [0.0, 0.0, 1.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=10)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_first_derivatives[index],
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
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_first_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [-0.853553, 0.146447, 0.5],
            [-0.736124, 0.256932, 0.34866],
            [-0.619536, 0.352686, 0.227975],
            [-0.503791, 0.433709, 0.137944],
            [-0.388889, 0.5, 0.0785674],
            [-0.274829, 0.55156, 0.0498449],
            [-0.161612, 0.588388, 0.0517767],
            [-0.0492368, 0.610485, 0.0843627],
            [0.0622956, 0.617851, 0.147603],
            [0.172985, 0.610485, 0.241498],
            [0.282833, 0.588388, 0.366046],
            [0.391838, 0.55156, 0.521249],
            [0.5, 0.5, 0.707107],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_first_derivatives[index],
            err_msg=(
                "Fails to find the first_derivative along a "
                "multi-displacement non-axis-aligned b-spline."
            ),
        )


def test_non_unit_length_bspline_first_derivative() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_first_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [6.0, 0.0, 0.0],
            [5.04167, 0.458333, 0.0208333],
            [4.16667, 0.833333, 0.0833333],
            [3.375, 1.125, 0.1875],
            [2.66667, 1.33333, 0.333333],
            [2.04167, 1.45833, 0.520833],
            [1.5, 1.5, 0.75],
            [1.04167, 1.45833, 1.02083],
            [0.666667, 1.33333, 1.33333],
            [0.375, 1.125, 1.6875],
            [0.166667, 0.833333, 2.08333],
            [0.0416667, 0.458333, 2.52083],
            [0.0, 0.0, 3.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_first_derivatives[index],
            err_msg=(
                "Fails to find the first_derivative along a "
                "multi-displacement b-spline with non-unit length."
            ),
        )


def test_repeated_control_point_bspline_first_derivative() -> None:
    """Tests that a b-spline with a repeated control point will have its
    first_derivative at that control point for 1.0 parameter distance.
    """
    # a b-spline with a repeated control point and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_first_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.04167, 0.458333, 0.0208333],
            [4.16667, 0.833333, 0.0833333],
            [3.375, 1.125, 0.1875],
            [2.66667, 1.33333, 0.333333],
            [2.04167, 1.45833, 0.520833],
            [1.5, 1.5, 0.75],
            [1.04167, 1.45833, 1.02083],
            [0.666667, 1.33333, 1.33333],
            [0.375, 1.125, 1.6875],
            [0.166667, 0.833333, 2.08333],
            [0.0416667, 0.458333, 2.52083],
            [0.0, 0.0, 3.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 4.0, num=17)):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            nominal_first_derivatives[index],
            err_msg=(
                "Fails to find the first_derivative along a b-spline "
                "with a repeated control point."
            ),
        )


# second derivative


def test_single_point_bspline_second_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_second_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_axis_bspline_second_derivative() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        second_derivative = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [1.0, 0.0, 0.0],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {second_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_two_axis_bspline_second_derivative() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        second_derivative = parameter * control_points[1]
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [2.0 ** -0.5, 2.0 ** -0.5, 0.0],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {second_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_three_axis_bspline_second_derivative() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct second_derivative vector.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        second_derivative = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            [3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {second_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_second_derivative() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct second derivative vector.
    """
    # a length=1.0 curve with displacements along each axis.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_second_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=10) (inputs scaled by 1/3)
        [
            [-2.0, 2.0, 0.0],
            [-1.77778, 1.55556, 0.222222],
            [-1.55556, 1.11111, 0.444444],
            [-1.33333, 0.666667, 0.666667],
            [-1.11111, 0.222222, 0.888889],
            [-0.888889, -0.222222, 1.11111],
            [-0.666667, -0.666667, 1.33333],
            [-0.444444, -1.11111, 1.55556],
            [-0.222222, -1.55556, 1.77778],
            [0.0, -2.0, 2.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=10)):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            nominal_second_derivatives[index],
            err_msg=(
                "Fails to find the second_derivative along a "
                "multi-displacement bspline."
            ),
        )


def test_multi_segment_non_axis_aligned_bspline_second_derivative() -> None:
    """Tests that a multi-displacement b-spline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment b-spline that isn't axis aligned.  Each
    # segment is length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_second_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [1.41421, 1.41421, -2.0],
            [1.4041, 1.23744, -1.63215],
            [1.39399, 1.06066, -1.2643],
            [1.38388, 0.883883, -0.896447],
            [1.37377, 0.707107, -0.528595],
            [1.36366, 0.53033, -0.160744],
            [1.35355, 0.353553, 0.207107],
            [1.34344, 0.176777, 0.574958],
            [1.33333, 4.44089 * 10 ** -16, 0.942809],
            [1.32322, -0.176777, 1.31066],
            [1.31311, -0.353553, 1.67851],
            [1.303, -0.53033, 2.04636],
            [1.29289, -0.707107, 2.41421],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            nominal_second_derivatives[index],
            err_msg=(
                "Fails to find the second_derivative along a "
                "multi-displacement non-axis-aligned b-spline."
            ),
        )


def test_non_unit_length_bspline_second_derivative() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_second_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [-12.0, 6.0, 0.0],
            [-11.0, 5.0, 0.5],
            [-10.0, 4.0, 1.0],
            [-9.0, 3.0, 1.5],
            [-8.0, 2.0, 2.0],
            [-7.0, 1.0, 2.5],
            [-6.0, 0.0, 3.0],
            [-5.0, -1.0, 3.5],
            [-4.0, -2.0, 4.0],
            [-3.0, -3.0, 4.5],
            [-2.0, -4.0, 5.0],
            [-1.0, -5.0, 5.5],
            [0.0, -6.0, 6.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 3.0, num=13)):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            nominal_second_derivatives[index],
            err_msg=(
                "Fails to find the second_derivative along a "
                "multi-displacement b-spline with non-unit length."
            ),
        )


def test_repeated_control_point_bspline_second_derivative() -> None:
    """Tests that a b-spline with a repeated control point will have its
    second_derivative at that control point for 1.0 parameter distance.
    """
    # a b-spline with a repeated control point and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_second_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 3.0, num=13) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-12.0, 6.0, 0.0],
            [-11.0, 5.0, 0.5],
            [-10.0, 4.0, 1.0],
            [-9.0, 3.0, 1.5],
            [-8.0, 2.0, 2.0],
            [-7.0, 1.0, 2.5],
            [-6.0, 0.0, 3.0],
            [-5.0, -1.0, 3.5],
            [-4.0, -2.0, 4.0],
            [-3.0, -3.0, 4.5],
            [-2.0, -4.0, 5.0],
            [-1.0, -5.0, 5.5],
            [0.0, -6.0, 6.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 4.0, num=17)):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            nominal_second_derivatives[index],
            err_msg=(
                "Fails to find the second_derivative along a b-spline "
                "with a repeated control point."
            ),
        )


# third derivative


def test_single_point_bspline_third_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_point_not_origin_bspline_third_derivative() -> None:
    """Tests that along the entire line, a b-spline defined by a single
    point is equal to just that single point, where that point is not
    the origin.
    """
    control_points = np.array([(0.0, 0.0, 0.0)])
    knot_vector = np.array([0, 0, 1, 1])
    curve = BSplineCurve(control_points, knot_vector)
    for parameter in np.linspace(0.0, 1.0, num=5):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [0, 0, 0],
            err_msg=(
                f"Fails to say that a b-spline defined by "
                f"{control_points} is equal to {control_points[0]} at "
                f"parameter {parameter}."
            ),
        )


def test_single_axis_bspline_third_derivative() -> None:
    """Tests that a bspline that only varies along a single axis works."""
    control_points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        third_derivative = (parameter, 0.0, 0.0)
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [1.0, 0.0, 0.0],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {third_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_two_axis_bspline_third_derivative() -> None:
    """Tests that a bspline that varies in a coordinate plane works."""
    # a length=1.0 curve in 2D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (2.0 ** -0.5, 2.0 ** -0.5, 0.0)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        third_derivative = parameter * control_points[1]
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [2.0 ** -0.5, 2.0 ** -0.5, 0.0],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {third_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_three_axis_bspline_third_derivative() -> None:
    """Tests that a simple bspline that varies in a 3D coordinate
    system has the correct third_derivative vector.
    """
    # a length=1.0 curve in 3D
    control_points = np.array(
        [(0.0, 0.0, 0.0), (3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5)]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 1.0, num=5):
        third_derivative = tuple(parameter * (3.0 ** -0.5) for _ in range(3))
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            [3.0 ** -0.5, 3.0 ** -0.5, 3.0 ** -0.5],
            err_msg=(
                f"Fails to say that a bspline defined by "
                f"{control_points} is equal to {third_derivative} at "
                f"parameter {parameter}."
            ),
        )


def test_multi_displacement_bspline_third_derivative() -> None:
    """Tests that a bspline made from multiple displacements has the
    correct third derivative vector.
    """
    # a length=1.0 curve with displacements along each axis.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0 / 3.0, 0.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 3.0, num=10):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            # computed using Mathematica's BSplineFunction
            [2.0, -4.0, 2.0],
            err_msg=(
                "Fails to find the third_derivative along a "
                "multi-displacement bspline."
            ),
        )


def test_multi_segment_non_axis_aligned_bspline_third_derivative() -> None:
    """Tests that a multi-displacement b-spline with non-axis-aligned
    segments works.
    """
    # a length=1.0 3-segment b-spline that isn't axis aligned.  Each
    # segment is length=1/3.
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (-1 / 12 * (2 + 2 ** 0.5), 1 / 12 * (2 - 2 ** 0.5), 1 / 6),
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),
            (-1.0 / 6.0, 1.0 / 2.0, 1.0 / (3.0 * (2 ** 0.5))),
        ]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 3.0, num=13):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            # computed using Mathematica's BSplineFunction over
            [-0.12132, -2.12132, 4.41421],
            err_msg=(
                "Fails to find the third derivative along a "
                "multi-displacement non-axis-aligned b-spline."
            ),
        )


def test_non_unit_length_bspline_third_derivative() -> None:
    """Tests that a non-unit length b-spline has its parameter go up to
    the number of segments of the b-spline.
    """
    # a multi-displacement b-spline with length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    for parameter in np.linspace(0.0, 3.0, num=13):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            # computed using Mathematica's BSplineFunction over
            [12.0, -12.0, 6.0],
            err_msg=(
                "Fails to find the third_derivative along a "
                "multi-displacement b-spline with non-unit length."
            ),
        )


def test_repeated_control_point_bspline_third_derivative() -> None:
    """Tests that a b-spline with a repeated control point will have its
    third_derivative at that control point for 1.0 parameter distance.
    """
    # a b-spline with a repeated control point and length=4.0
    control_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 1.0),
        ]
    )
    curve = BSplineCurve(control_points)
    nominal_third_derivatives = np.array(
        # computed using Mathematica's BSplineFunction over
        # np.linspace(0.0, 4.0, num=17) (inputs scaled by 1/3)
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
            [12.0, -12.0, 6.0],
        ]
    )
    for index, parameter in np.ndenumerate(np.linspace(0.0, 4.0, num=17)):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            nominal_third_derivatives[index],
            err_msg=(
                "Fails to find the third derivative along a b-spline "
                "with a repeated control point."
            ),
        )
