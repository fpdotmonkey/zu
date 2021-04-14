"""
Unit tests to verify that analytic curves are well-behaved.
"""

import logging

import numpy as np

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_constant_curve_radius() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has radius at
    that constant.
    """
    position = [0.0, 0.0, 0.0]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """First derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == constant_curve_radius(
            parameter
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == linear_curve_radius(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == quadratic_curve_radius(
            parameter
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
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == constant_curve_radius(
            parameter
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == linear_curve_radius(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == quadratic_curve_radius(
            parameter
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

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.radius_at(parameter) == quadratic_curve_radius(
            parameter
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
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == constant_curve_first_derivative(parameter), (
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == linear_curve_first_derivative(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == quadratic_curve_first_derivative(parameter), (
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
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == constant_curve_first_derivative(parameter), (
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == linear_curve_first_derivative(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == quadratic_curve_first_derivative(parameter), (
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

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.first_derivative_at(
            parameter
        ) == quadratic_curve_first_derivative(parameter), (
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
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == constant_curve_second_derivative(parameter), (
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == linear_curve_second_derivative(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == quadratic_curve_second_derivative(parameter), (
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
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == constant_curve_second_derivative(parameter), (
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
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == linear_curve_second_derivative(parameter), (
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
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == quadratic_curve_second_derivative(parameter), (
            "Fails to say that the second derivative of a "
            "constant-acceleration curve defined over all real "
            f"parameters is equal to its acceleration {acceleration}."
        )


def test_non_linear_curve_first_derivative() -> None:
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

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.second_derivative_at(
            parameter
        ) == quadratic_curve_second_derivative(parameter), (
            "Fails to say that a non-linear curve defined over all real "
            "parameters has the correct second derivative at parameter "
            f"{parameter}."
        )


# tangent vector


def test_constant_curve_tangent_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has tangent of
    [0, 0, 0].
    """
    position = [0.0, 0.0, 0.0]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector(parameter), [0.0, 0.0, 0.0]
        ), (
            "Fails to say that a constant curve defined over all real "
            f"parameters has tangent vector of [0, 0, 0]."
        )


def test_linear_single_axis_curve_tangent_vector() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct tangent vector.
    """

    velocity = [1.0, 0.0, 0.0]

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(curve.tangent_vector(parameter), [1, 0, 0]), (
            "Fails to say that the tangent vector of a straight line is "
            "along that straight line."
        )


def test_quadratic_single_axis_curve_tangent_vector() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct radius.
    """

    acceleration = [1.0, 0.0, 0.0]

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector(parameter), acceleration
        ), "Fails to say that the tangent vector of a curve with "


def test_constant_not_origin_curve_tangent_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has tangent vector
    of [0, 0, 0].
    """
    position = [1.61, -2.71, 3.14]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """First derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.is_equal(curve.tangent_vector(parameter), [0.0, 0.0, 0.0]), (
            "Fails to say that the tangent vector of a constant curve "
            "not on the origin defined over all real parameters is "
            "equal to [0, 0, 0]."
        )


def test_linear_off_axis_curve_tangent_vector() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct tangent vector.
    """

    velocity = [1.61, -2.71, 3.14]

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            # calculated as velocity / sqrt(dot(velocity, velocity))
            np.array([0.36185897, -0.60909181, 0.70573738]),
        ), (
            "Fails to say that the tangent vector of a linear curve "
            "defined over all real parameters is equal to "
            "[0.36185897, -0.60909181, 0.70573738]."
        )


def test_quadratic_off_axis_curve_tangent_vector() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct tangent vector.
    """

    acceleration = [1.61, -2.71, 3.14]

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector(parameter),
            # calculated as
            # parameter * acceleration
            # / sqrt(dot(parameter * acceleration, parameter * acceleration))
            np.array([0.36185897, -0.60909181, 0.70573738]),
        ), (
            "Fails to say that the tangent vector of a "
            "constant-acceleration curve defined over all real "
            "parameters is equal to "
            "[0.36185897, -0.60909181, 0.70573738]."
        )


def test_non_linear_curve_tangent_vector() -> None:
    """Test that a non-linear curve gives the correct tangent vector."""

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector(parameter),
            np.sqrt(0.5) * curve_first_derivative(parameter),
        ), (
            "Fails to say that the a non-linear curve defined over all "
            "real parameters has the correct tangent vector at "
            f"parameter {parameter}."
        )


# normal vector


def test_constant_curve_normal_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has normal of
    [0, 0, 0].
    """
    position = [0.0, 0.0, 0.0]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector(parameter), [0.0, 0.0, 0.0]
        ), (
            "Fails to say that a constant curve defined over all real "
            f"parameters has normal vector of [0, 0, 0]."
        )


def test_linear_single_axis_curve_normal_vector() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct normal vector.
    """

    velocity = [1.0, 0.0, 0.0]

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert (
            np.dot(
                curve.tangent_vector(parameter), curve.normal_vector(parameter)
            )
            == 0.0
        ), (
            "Fails to say that the normal vector of a straight line is "
            "along that straight line."
        )


def test_quadratic_single_axis_curve_normal_vector() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct radius.
    """

    acceleration = [1.0, 0.0, 0.0]

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector(parameter), acceleration
        ), "Fails to say that the normal vector of a curve with "


def test_constant_not_origin_curve_normal_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has normal vector
    of [0, 0, 0].
    """
    position = [1.61, -2.71, 3.14]

    def constant_curve_radius(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """First derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.is_equal(curve.normal_vector(parameter), [0.0, 0.0, 0.0]), (
            "Fails to say that the normal vector of a constant curve "
            "not on the origin defined over all real parameters is "
            "equal to [0, 0, 0]."
        )


def test_linear_off_axis_curve_normal_vector() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct normal vector.
    """

    velocity = [1.61, -2.71, 3.14]

    def linear_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return parameter * np.array(velocity)

    def linear_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a linear curve."""
        return np.array(velocity)

    def linear_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.radius_at(parameter),
            # calculated as velocity / sqrt(dot(velocity, velocity))
            np.array([0.36185897, -0.60909181, 0.70573738]),
        ), (
            "Fails to say that the normal vector of a linear curve "
            "defined over all real parameters is equal to "
            "[0.36185897, -0.60909181, 0.70573738]."
        )


def test_quadratic_off_axis_curve_normal_vector() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct normal vector.
    """

    acceleration = [1.61, -2.71, 3.14]

    def quadratic_curve_radius(parameter: float) -> npt.ArrayLike:
        """Linear curve."""
        return 0.5 * parameter * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector(parameter),
            # calculated as
            # parameter * acceleration
            # / sqrt(dot(parameter * acceleration, parameter * acceleration))
            np.array([0.36185897, -0.60909181, 0.70573738]),
        ), (
            "Fails to say that the normal vector of a "
            "constant-acceleration curve defined over all real "
            "parameters is equal to "
            "[0.36185897, -0.60909181, 0.70573738]."
        )


def test_non_linear_curve_normal_vector() -> None:
    """Test that a non-linear curve gives the correct normal vector."""

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Helix radius."""
        return np.array([np.cos(parameter), np.sin(parameter), parameter])

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Helix first derivative."""
        return np.array([-np.sin(parameter), np.cos(parameter), 1.0])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Helix second derivative."""
        return np.array([-np.cos(parameter), -np.sin(parameter), 0.0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector(parameter),
            np.sqrt(0.5) * curve_first_derivative(parameter),
        ), (
            "Fails to say that the a non-linear curve defined over all "
            "real parameters has the correct normal vector at parameter "
            f"{parameter}."
        )
