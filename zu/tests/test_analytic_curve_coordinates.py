"""
Unit tests to verify that analytic curves are well-behaved.
"""

# pylint: disable=too-many-lines

import logging

import numpy as np

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


# radius


def test_constant_curve_radius() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has radius at
    that constant.
    """
    position = [0.0, 0.0, 0.0]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """First derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), constant_curve_radius(parameter)
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), linear_curve_radius(parameter)
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), quadratic_curve_radius(parameter)
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), constant_curve_radius(parameter)
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), linear_curve_radius(parameter)
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), quadratic_curve_radius(parameter)
        ), (
            "Fails to say that the radius of a constant-acceleration "
            "curve defined over all real parameters is equal to its "
            f"acceleration {acceleration} times 0.5 time parameter "
            f"{parameter} ** 2.0."
        )


def test_non_linear_curve_radius() -> None:
    """Test that a non-linear curve gives the correct radius."""

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    def curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([np.sin(parameter), -np.cos(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
        curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter), curve_radius(parameter)
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            constant_curve_first_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            linear_curve_first_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            quadratic_curve_first_derivative(parameter),
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            constant_curve_first_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            linear_curve_first_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            quadratic_curve_first_derivative(parameter),
        ), (
            "Fails to say that a constant-acceleration curve defined "
            "over all real parameters is equal to its acceleration "
            f"{acceleration} times its parameter {parameter}."
        )


def test_non_linear_curve_first_derivative() -> None:
    """Test that a non-linear curve gives the correct first derivative."""

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    def curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([np.sin(parameter), -np.cos(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
        curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.first_derivative_at(parameter),
            curve_first_derivative(parameter),
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            constant_curve_second_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            linear_curve_second_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            quadratic_curve_second_derivative(parameter),
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            constant_curve_second_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            linear_curve_second_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            quadratic_curve_second_derivative(parameter),
        ), (
            "Fails to say that the second derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_non_linear_curve_second_derivative() -> None:
    """Test that a non-linear curve gives the correct second
    derivative.
    """

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    def curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Helix third derivative."""
        return np.array([np.sin(parameter), -np.cos(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
        curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            curve_second_derivative(parameter),
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.second_derivative_at(parameter),
            constant_curve_second_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            linear_curve_third_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            quadratic_curve_third_derivative(parameter),
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

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        del parameter
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            constant_curve_third_derivative(parameter),
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

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        del parameter
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            linear_curve_third_derivative(parameter),
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

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        del parameter
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        del parameter
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            quadratic_curve_third_derivative(parameter),
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

    def cubic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Cubic curve."""
        return 1.0 / 6.0 * (parameter ** 3) * np.array(jerk)

    def cubic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a cubic curve."""
        return 0.5 * (parameter ** 2) * np.array(jerk)

    def cubic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a cubic curve."""
        return parameter * np.array(jerk)

    def cubic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a cubic curve."""
        del parameter
        return np.array(jerk)

    curve = AnalyticCurve(
        cubic_curve_radius,
        cubic_curve_first_derivative,
        cubic_curve_second_derivative,
        cubic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            cubic_curve_third_derivative(parameter),
        ), (
            "Fails to say that the third derivative of a "
            "constant-jerk curve defined over all real parameters is "
            f"equal to its jerk {jerk}."
        )


def test_non_linear_curve_third_derivative() -> None:
    """Test that a non-linear curve gives the correct third
    derivative.
    """

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    def curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Helix third derivative."""
        return np.array([np.sin(parameter), -np.cos(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
        curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.third_derivative_at(parameter),
            curve_third_derivative(parameter),
        ), (
            "Fails to say that a non-linear curve defined over all real "
            "parameters has the correct third derivative at parameter "
            f"{parameter}."
        )
