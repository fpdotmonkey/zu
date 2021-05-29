"""
Unit tests to verify that analytic curves are well-behaved.
"""

import logging

import numpy as np
import pytest

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


# not enough coordinates


def test_not_enough_coordinates_leads_to_exception() -> None:
    """If not enough coordinates are given to do all the calculating,
    then an exception should be raised.
    """
    with pytest.raises(ValueError):
        AnalyticCurve((lambda parameter: np.array([0, 0, 0]),))  # type: ignore


# radius


def test_constant_curve_radius() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has radius at
    that constant.
    """
    position = [0.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), np.array(position)
        ), (
            "Fails to say that a constant curve defined over all real "
            f"parameters is equal to that constant at {parameter}."
        )


def test_linear_single_axis_curve_radius() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct radius.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), parameter * np.array(velocity)
        ), (
            "Fails to say that the radius of a linear curve defined "
            "over all real parameters is equal to its velocity "
            f"{velocity} times the parameter {parameter}."
        )


def test_quadratic_single_axis_curve_radius() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct radius.
    """
    acceleration = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            0.5 * (parameter ** 2) * np.array(acceleration),
        ), (
            "Fails to say that the radius of a constant-acceleration "
            "curve defined over all real parameters is equal to its "
            f"acceleration {acceleration} times 0.5 time parameter "
            f"{parameter} ** 2.0."
        )


def test_constant_not_origin_curve_radius() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has radius at
    that constant which isn't the origin.
    """
    position = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            np.array(position),
        ), (
            "Fails to say that the radius of a constant curve not on "
            "the origin defined over all real parameters is equal to "
            f"that constant at {parameter}."
        )


def test_linear_off_axis_curve_radius() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct radius.
    """
    velocity = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            parameter * np.array(velocity),
        ), (
            "Fails to say that the radius of a linear curve defined "
            "over all real parameters is equal to its velocity "
            f"{velocity} times the parameter {parameter}."
        )


def test_quadratic_off_axis_curve_radius() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct radius.
    """
    acceleration = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            0.5 * (parameter ** 2) * np.array(acceleration),
        ), (
            "Fails to say that the radius of a constant-acceleration "
            "curve defined over all real parameters is equal to its "
            f"acceleration {acceleration} times 0.5 time parameter "
            f"{parameter} ** 2.0."
        )


def test_non_linear_curve_radius() -> None:
    """Test that a non-linear curve gives the correct radius."""

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(
                [np.cos(parameter), np.sin(parameter), parameter]
            ),
            lambda parameter: np.array(
                [-np.sin(parameter), np.cos(parameter), 1.0]
            ),
            lambda parameter: np.array(
                [-np.cos(parameter), -np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [np.sin(parameter), -np.cos(parameter), 0.0]
            ),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            np.array([np.cos(parameter), np.sin(parameter), parameter]),
        ), (
            "Fails to say that the a non-linear curve defined over all "
            "real parameters has the correct radius at parameter "
            f"{parameter}."
        )


# first derivative


def test_constant_curve_first_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has first
    derivative of zero.
    """
    position = [0.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the first derivative of a constant curve "
            "defined over all real parameters is equal to zero."
        )


def test_linear_single_axis_curve_first_derivative() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has constant velocity.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            np.array(velocity),
        ), (
            "Fails to say that the first derivative of a linear curve "
            "defined over all real parameters is equal to its velocity "
            f"{velocity}."
        )


def test_quadratic_single_axis_curve_first_derivative() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct first derivative.
    """
    acceleration = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            parameter * np.array(acceleration),
        ), (
            "Fails to say that the first derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration} "
            f"times its parameter {parameter}"
        )


def test_constant_not_origin_curve_first_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has first
    derivative of zero.
    """
    position = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the first derivative of a constant curve "
            "not on the origin defined over all real parameters is "
            f"equal to that constant at {parameter}."
        )


def test_linear_off_axis_curve_first_derivative() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has constant first derivative.
    """
    velocity = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            np.array(velocity),
        ), (
            "Fails to say that the first derivative of a linear curve "
            "defined over all real parameters is equal to its velocity "
            f"{velocity}."
        )


def test_quadratic_off_axis_curve_first_derivative() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct first derivative.
    """
    acceleration = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            parameter * np.array(acceleration),
        ), (
            "Fails to say that a constant-acceleration curve defined "
            "over all real parameters is equal to its acceleration "
            f"{acceleration} times its parameter {parameter}."
        )


def test_non_linear_curve_first_derivative() -> None:
    """Test that a non-linear curve gives the correct first derivative."""
    curve = AnalyticCurve(
        (
            lambda parameter: np.array(
                [np.cos(parameter), np.sin(parameter), parameter]
            ),
            lambda parameter: np.array(
                [-np.sin(parameter), np.cos(parameter), 1.0]
            ),
            lambda parameter: np.array(
                [-np.cos(parameter), -np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [np.sin(parameter), -np.cos(parameter), 0.0]
            ),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            np.array([-np.sin(parameter), np.cos(parameter), 1.0]),
        ), (
            "Fails to say that a non-linear curve defined over all real "
            "parameters has the correct first derivative at parameter "
            f"{parameter}."
        )


# second derivative


def test_constant_curve_second_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has second
    derivative of zero.
    """
    position = [0.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the second derivative of a constant "
            "curve defined over all real parameters is equal to zero."
        )


def test_linear_single_axis_curve_second_derivative() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has second derivative of zero.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the second derivative of a linear curve "
            "defined over all real parameters is equal to zero."
        )


def test_quadratic_single_axis_curve_second_derivative() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has constant second derivative.
    """
    acceleration = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array(acceleration),
        ), (
            "Fails to say that the second derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_constant_not_origin_curve_second_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has second
    derivative of zero.
    """
    position = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the second derivative of a constant "
            "curve not on the origin defined over all real parameters "
            "is equal to zero."
        )


def test_linear_off_axis_curve_second_derivative() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has zero second derivative.
    """
    velocity = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the second derivative of a linear curve "
            "defined over all real parameters is equal to zero."
        )


def test_quadratic_off_axis_curve_second_derivative() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has constant second derivative.
    """
    acceleration = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array(acceleration),
        ), (
            "Fails to say that the second derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_non_linear_curve_second_derivative() -> None:
    """Test that a non-linear curve gives the correct second
    derivative.
    """
    curve = AnalyticCurve(
        (
            lambda parameter: np.array(
                [np.cos(parameter), np.sin(parameter), parameter]
            ),
            lambda parameter: np.array(
                [-np.sin(parameter), np.cos(parameter), 1.0]
            ),
            lambda parameter: np.array(
                [-np.cos(parameter), -np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [np.sin(parameter), -np.cos(parameter), 0.0]
            ),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([-np.cos(parameter), -np.sin(parameter), 0.0]),
        ), (
            "Fails to say that a non-linear curve defined over all real "
            "parameters has the correct second derivative at parameter "
            f"{parameter}."
        )


# third derivative


def test_constant_curve_third_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has third
    derivative of zero.
    """
    position = [0.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a constant "
            "curve defined over all real parameters is equal to zero."
        )


def test_linear_single_axis_curve_third_derivative() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has third derivative of zero.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a linear curve "
            "defined over all real parameters is equal to zero."
        )


def test_quadratic_single_axis_curve_third_derivative() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has constant third derivative.
    """
    acceleration = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_constant_not_origin_curve_third_derivative() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has third
    derivative of zero.
    """
    position = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: np.array(position),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a constant "
            "curve not on the origin defined over all real parameters "
            "is equal to zero."
        )


def test_linear_off_axis_curve_third_derivative() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has zero third derivative.
    """
    velocity = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a linear curve "
            "defined over all real parameters is equal to zero."
        )


def test_quadratic_off_axis_curve_third_derivative() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has constant third derivative.
    """
    acceleration = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: 0.5 * (parameter ** 2) * np.array(acceleration),
            lambda parameter: parameter * np.array(acceleration),
            lambda parameter: np.array(acceleration),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([0.0, 0.0, 0.0]),
        ), (
            "Fails to say that the third derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_cubic_off_axis_curve_third_derivative() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has constant third derivative.
    """
    jerk = [1.61, -2.71, 3.14]

    curve = AnalyticCurve(
        (
            lambda parameter: 1.0 / 6.0 * (parameter ** 3) * np.array(jerk),
            lambda parameter: 0.5 * (parameter ** 2) * np.array(jerk),
            lambda parameter: parameter * np.array(jerk),
            lambda parameter: np.array(jerk),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array(jerk),
        ), (
            "Fails to say that the third derivative of a "
            "constant-jerk curve defined over all real parameters is "
            f"equal to its jerk {jerk}."
        )


def test_non_linear_curve_third_derivative() -> None:
    """Test that a non-linear curve gives the correct third
    derivative.
    """
    curve = AnalyticCurve(
        (
            lambda parameter: np.array(
                [np.cos(parameter), np.sin(parameter), parameter]
            ),
            lambda parameter: np.array(
                [-np.sin(parameter), np.cos(parameter), 1.0]
            ),
            lambda parameter: np.array(
                [-np.cos(parameter), -np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [np.sin(parameter), -np.cos(parameter), 0.0]
            ),
        )
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            np.array([np.sin(parameter), -np.cos(parameter), 0.0]),
        ), (
            "Fails to say that a non-linear curve defined over all real "
            "parameters has the correct third derivative at parameter "
            f"{parameter}."
        )
