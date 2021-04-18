"""
Tests that an analytic curve can have some finite length defined by an
upper or a lower limit or both.
"""


import logging

import numpy as np
import pytest

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_inifinite_curve() -> None:
    """If the keyword arguments `upper_bound=None, lower_bound=None` are
    given, then the curve should extend off to +/- infinity, as normal.
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
        upper_bound=None,
        lower_bound=None,
    )

    for parameter in np.append(
        -np.power(10, np.linspace(0, 20, num=21)),
        np.power(10, np.linspace(0, 20, num=21)),
    ):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            np.array([parameter, 0, 0]),
            err_msg="Fails to generate an infinitely long curve when "
            "upper_bound and lower_bound are set to None.",
        )


def test_curve_with_upper_bound() -> None:
    """A curve where the keyword argument `upper_bound=[float]` is
    supplied should result in a curve where the input parameter can be
    in the interval (-inf, upper_bound].
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
        upper_bound=10.0,
        lower_bound=None,
    )

    for parameter in np.append(
        np.geomspace(-4e6, -1, num=10), np.linspace(0, 10, num=3)
    ):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            np.array([parameter, 0, 0]),
            err_msg="Fails to generate a curve where parameters in the "
            "interval (-inf, upper_bound] are valid.",
        )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.radius_at(10.0001)
    assert above_bounds.value[0] and isinstance(
        above_bounds.value[0], str
    ), "Fails to send the AboveBounds exception with a message."
    assert np.isclose(
        above_bounds.value[1], 10.0
    ), "Fails to send the upper bound with the AboveBounds exception."


def test_curve_with_lower_bound() -> None:
    """A curve where the keyword argument `upper_bound=[float]` is
    supplied should result in a curve where the input parameter can be
    in the interval [lower_bound, inf).
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
        upper_bound=None,
        lower_bound=-10.0,
    )

    for parameter in np.append(
        np.linspace(-10.0, 0, num=10), np.linspace(1, 4e6, num=3)
    ):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            np.array([parameter, 0, 0]),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, inf) are valid.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.radius_at(-10.0001)
    assert below_bounds.value[0] and isinstance(
        below_bounds.value[0], str
    ), "Fails to send the BelowBounds exception with a message."
    assert np.isclose(
        below_bounds.value[1], -10.0
    ), "Fails to send the lower bound with the BelowBounds exception."


def test_curve_with_upper_and_lower_bounds() -> None:
    """A curve where the keyword argument `upper_bound=[float]` is
    supplied should result in a curve where the input parameter can be
    in the interval [lower_bound, inf).
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
        upper_bound=10.0,
        lower_bound=-10.0,
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            np.array([parameter, 0, 0]),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.radius_at(-10.0001)
    assert below_bounds.value[0] and isinstance(
        below_bounds.value[0], str
    ), "Fails to send the BelowBounds exception with a message."
    assert np.isclose(
        below_bounds.value[1], -10.0
    ), "Fails to send the lower bound with the BelowBounds exception."

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.radius_at(10.0001)
    assert above_bounds.value[0] and isinstance(
        above_bounds.value[0], str
    ), "Fails to send the AboveBounds exception with a message."
    assert np.isclose(
        above_bounds.value[1], 10.0
    ), "Fails to send the upper bound with the AboveBounds exception."
