"""
Tests to validate that the moving trihedral of an analytic curve (i.e.
its tangent, normal, and binormal vectors) are correct.  There are also
tests for the curvature and torsion of the curves.
"""

import logging

import numpy as np

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector_at(parameter), [0.0, 0.0, 0.0]
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(curve.tangent_vector_at(parameter), [1, 0, 0]), (
            "Fails to say that the tangent vector of a straight line is "
            "along that straight line."
        )


def test_quadratic_single_axis_curve_tangent_vector() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct tangent vector.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector_at(parameter), acceleration
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.is_equal(
            curve.tangent_vector_at(parameter), [0.0, 0.0, 0.0]
        ), (
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector_at(parameter),
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
        return 0.5 * (parameter ** 2) * np.array(acceleration)

    def quadratic_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a quadratic curve."""
        return parameter * np.array(acceleration)

    def quadratic_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a quadratic curve."""
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.tangent_vector_at(parameter),
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
            curve.tangent_vector_at(parameter),
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector_at(parameter), [0.0, 0.0, 0.0]
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent is as
        # good a test as we can get.  It'd be good to come up with a
        # reasonable convention for what the normal vector should be so
        # that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.normal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the normal vector of a straight line is "
            "perpendicular to the tangent vector at that point."
        )


def test_quadratic_single_axis_curve_normal_vector() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct normal vector.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent is as
        # good a test as we can get.  It'd be good to come up with a
        # reasonable convention for what the normal vector should be so
        # that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.normal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the normal vector of a straight-line "
            "curve with constantly accelerating parameter is "
            "perpendicular to the tangent vector"
        )


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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.normal_vector_at(parameter), [0.0, 0.0, 0.0]
        ), (
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent is as
        # good a test as we can get.  It'd be good to come up with a
        # reasonable convention for what the normal vector should be so
        # that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.normal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the normal vector of a linear curve "
            "defined over all real parameters is perpendicular to the "
            "tangent vector."
        )


def test_quadratic_off_axis_curve_normal_vector() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct normal vector.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent is as
        # good a test as we can get.  It'd be good to come up with a
        # reasonable convention for what the normal vector should be so
        # that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.normal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the normal vector of a "
            "constant-acceleration curve defined over all real "
            "parameters is perpendicular to the tangent vector."
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
        normal_vector = np.array([-np.cos(t), -np.sin(t), 0.0])
        assert np.array_equal(
            curve.normal_vector_at(parameter),
            normal_vector,
        ), (
            "Fails to say that the a non-linear curve defined over all "
            "real parameters has the correct normal vector of "
            f"{normal_vector} at parameter {parameter}."
        )


# binormal vector


def test_constant_curve_binormal_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has binormal of
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.binormal_vector_at(parameter), [0.0, 0.0, 0.0]
        ), (
            "Fails to say that a constant curve defined over all real "
            f"parameters has binormal vector of [0, 0, 0]."
        )


def test_linear_single_axis_curve_binormal_vector() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct binormal vector.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent and
        # normal is as good a test as we can get.  It'd be good to come
        # up with a reasonable convention for what the binormal vector
        # should be so that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ) and (
            np.dot(
                curve.normal_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the binormal vector of a straight line is "
            "perpendicular to both the tangent and normal vector at "
            "that point."
        )


def test_quadratic_single_axis_curve_binormal_vector() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct binormal vector.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent and
        # normal is as good a test as we can get.  It'd be good to come
        # up with a reasonable convention for what the binormal vector
        # should be so that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ) and (
            np.dot(
                curve.normal_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the binormal vector of a straight-line "
            "curve with constantly accelerating parameter is "
            "perpendicular to both the tangent vector and the normal "
            "vector."
        )


def test_constant_not_origin_curve_binormal_vector() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has binormal vector
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.array_equal(
            curve.binormal_vector_at(parameter), [0.0, 0.0, 0.0]
        ), (
            "Fails to say that the binormal vector of a constant curve "
            "not on the origin defined over all real parameters is "
            "equal to [0, 0, 0]."
        )


def test_linear_off_axis_curve_binormal_vector() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has the correct binormal vector.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent and
        # normal is as good a test as we can get.  It'd be good to come
        # up with a reasonable convention for what the binormal vector
        # should be so that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ) and (
            np.dot(
                curve.normal_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the binormal vector of a linear curve "
            "defined over all real parameters is perpendicular to the "
            "tangent vector."
        )


def test_quadratic_off_axis_curve_binormal_vector() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has the correct binormal vector.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        # since this is a straight line, perpendicular to tangent and
        # normal is as good a test as we can get.  It'd be good to come
        # up with a reasonable convention for what the binormal vector
        # should be so that it can be stable.
        assert (
            np.dot(
                curve.tangent_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ) and (
            np.dot(
                curve.normal_vector_at(parameter),
                curve.binormal_vector_at(parameter),
            )
            == 0.0
        ), (
            "Fails to say that the binormal vector of a "
            "constant-acceleration curve defined over all real "
            "parameters is perpendicular to the tangent vector."
        )


def test_non_linear_curve_binormal_vector() -> None:
    """Test that a non-linear curve gives the correct binormal vector."""

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
        magnitude = np.sqrt(
            1.0
            + np.abs(np.cos(parameter)) ** 2
            + np.abs(np.sin(parameter)) ** 2
        )
        binormal_vector = np.array([np.sin(t), np.cos(t), 1.0]) / magnitude
        assert np.array_equal(
            curve.binormal_vector_at(parameter),
            binormal_vector,
        ), (
            "Fails to say that the a non-linear curve defined over all "
            "real parameters has the correct binormal vector of "
            f"{binormal_vector} at parameter {parameter}."
        )


# curvature


def test_constant_curve_curvature() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has curvature of
    NaN.
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.isnan(curve.curvature_at(parameter))(
            "Fails to say that a constant curve defined over all real "
            "parameters has curvature of NaN."
        )


def test_linear_single_axis_curve_curvature() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has curvature of 0.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.curvature_at(parameter) == 0.0, (
            "Fails to say that the curvature of a straight-line curve "
            f"is 0 at parameter {parameter}."
        )


def test_quadratic_single_axis_curve_curvature() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has a curvature of 0.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.curvature_at(parameter) == 0.0, (
            "Fails to say that the curvature of a straight-line "
            f"accelerating curve is 0 at parameter {parameter}."
        )


def test_constant_not_origin_curve_curvature() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has binormal vector
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.isnan(curve.curvature_at(parameter))(
            "Fails to say that a constant curve defined over all real "
            "parameters has curvature of NaN."
        )


def test_linear_off_axis_curve_curvature() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has curvature of 0.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.curvature_at(parameter) == 0.0, (
            "Fails to say that the curvature of a straight-line curve "
            f"is 0 at parameter {parameter}."
        )


def test_quadratic_off_axis_curve_curvature() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has curvature of 0.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.curvature_at(parameter) == 0.0, (
            "Fails to say that the curvature of a straight-line "
            f"accelerating curve is 0 at parameter {parameter}."
        )


def test_non_linear_curve_curvature() -> None:
    """Test that a non-linear curve gives the correct binormal vector."""

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
        curvature = 1 / np.sqrt(2.0)
        assert curve.curvature_at(parameter) == curvature, (
            "Fails to say that the curvature of a helix at parameter "
            f"{parameter} is {curvature}."
        )


# torsion


def test_constant_curve_torsion() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has torsion of
    NaN.
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.isnan(curve.torsion_at(parameter))(
            "Fails to say that a constant curve defined over all real "
            "parameters has torsion of NaN."
        )


def test_linear_single_axis_curve_torsion() -> None:
    """Test that a curve that varies linearly over a single axis over
    the entire range of the parameter (which is defined for all reals)
    has torsion of 0.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.torsion_at(parameter) == 0.0, (
            "Fails to say that the torsion of a straight-line curve "
            f"is 0 at parameter {parameter}."
        )


def test_quadratic_single_axis_curve_torsion() -> None:
    """Test that a curve that varies quadratically over a single axis
    over the entire range of the parameter (which is defined for all
    reals) has a torsion of 0.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.torsion_at(parameter) == 0.0, (
            "Fails to say that the torsion of a straight-line "
            f"accelerating curve is 0 at parameter {parameter}."
        )


def test_constant_not_origin_curve_torsion() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has torsion of 0.
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

    def constant_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_radius,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
        constant_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert np.isnan(curve.torsion_at(parameter))(
            "Fails to say that a constant curve defined over all real "
            "parameters has torsion of NaN."
        )


def test_linear_off_axis_curve_torsion() -> None:
    """Test that a curve that varies linearly over some axis over
    the entire range of the parameter (which is defined for all reals)
    has torsion of 0.
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

    def linear_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a linear curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        linear_curve_radius,
        linear_curve_first_derivative,
        linear_curve_second_derivative,
        linear_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.torsion_at(parameter) == 0.0, (
            "Fails to say that the torsion of a straight-line curve "
            f"is 0 at parameter {parameter}."
        )


def test_quadratic_off_axis_curve_torsion() -> None:
    """Test that a curve that varies quadratically over some axis
    over the entire range of the parameter (which is defined for all
    reals) has torsion of 0.
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
        return np.array(acceleration)

    def quadratic_curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Third derivative of a quadratic curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        quadratic_curve_radius,
        quadratic_curve_first_derivative,
        quadratic_curve_second_derivative,
        quadratic_curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.torsion_at(parameter) == 0.0, (
            "Fails to say that the torsion of a straight-line "
            f"accelerating curve is 0 at parameter {parameter}."
        )


def test_non_linear_curve_torsion() -> None:
    """Test that a non-linear curve gives the correct torsion."""

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
        assert curve.torsion_at(parameter) == 0.5, (
            "Fails to say that the torsion of a helix at parameter "
            f"{parameter} is 0.5."
        )


def test_tschirnhausen_cubic() -> None:
    """Test that all the trihedral stuff works under a gauntlet of a
    cubic polynomial function.
    """

    def curve_radius(parameter: float) -> npt.ArrayLike:
        """Tschirnhausen radius."""
        return np.array(
            [3 - parameter ** 2, parameter * (3 - parameter ** 2), -parameter]
        )

    def curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Tschirnhausen first derivative."""
        return np.array([-2 * parameter, 3 - 3 * parameter ** 2, -1])

    def curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Tschirnhausen second derivative."""
        return np.array([-2, -6 * parameter, 0])

    def curve_third_derivative(parameter: float) -> npt.ArrayLike:
        """Tschirnhausen third derivative."""
        return np.array([0, -6, 0])

    curve = AnalyticCurve(
        curve_radius,
        curve_first_derivative,
        curve_second_derivative,
        curve_third_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        curvature = 2 * np.sqrt(
            (10 + 9 * (parameter ** 2) * (3 + parameter ** 2))
            / (10 - 14 * parameter ** 2 + 9 * parameter ** 4) ** 3
        )
        assert curve.curvature_at(parameter) == curvature, (
            "Fails to get the correct curvature of a Tschirnhausen "
            f"polynomial {curvature} at parameter {parameter}."
        )

        torsion = -3 / (10 + 9 * (parameter ** 2) * (3 + parameter ** 2))
        assert curve.torsion_at(parameter) == torsion, (
            "Fails to get the correct torsion of a Tschirnhausen "
            f"polynomial {torsion} at parameter {parameter}."
        )

        tangent_vector = np.array(
            [-2 * parameter, -3 * (parameter ** 2 - 1), -1]
        ) / (np.sqrt(10 - 14 * parameter ** 2 + 9 * parameter ** 4))
        assert np.array_equal(
            curve.tangent_vector_at(parameter), tangent_vector
        ), (
            "Fails to get the correct tangent vector of a Tschirnhausen "
            f"polynomial {tangent_vector} at parameter {parameter}."
        )

        normal_vector = np.array(
            [
                -10 + 9 * parameter ** 4,
                3 * parameter * (3 + 2 * parameter ** 2),
                parameter * (-7 + 9 * parameter ** 2),
            ]
        ) / np.sqrt(
            (10 - 14 * parameter ** 2 + 9 * parameter ** 4)
            * (10 + 9 * (parameter ** 2) * (3 + parameter ** 2))
        )
        assert np.array_equal(
            curve.normal_vector_at(parameter), normal_vector
        ), (
            "Fails to get the correct normal vector of a Tschirnhausen "
            f"polynomial {normal_vector} at parameter {parameter}."
        )

        binormal_vector = np.array(
            [-3 * parameter, 1, 3 * (1 + parameter ** 2)]
        ) / np.sqrt(10 + 9 * (parameter ** 2) * (3 + parameter ** 2))
        assert np.array_equal(
            curve.binormal_vector_at(parameter), binormal_vector
        ), (
            "Fails to get the correct binormal vector of a Tschirnhausen "
            f"polynomial {binormal_vector} at parameter {parameter}."
        )
