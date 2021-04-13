"""
Unit tests to verify that analytic curves are well-behaved.
"""

import logging

import numpy as np

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_constant_curve_position() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has position at
    that constant.
    """
    position = [0.0, 0.0, 0.0]

    def constant_curve_position(parameter: float) -> npt.ArrayLike:
        """Constant curve at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_position,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.position_at(parameter) == np.array(position), (
            "Fails to say that a constant curve defined over all real "
            f"parameters is equal to that constant at {parameter}."
        )


def test_constant_curve_position_not_origin() -> None:
    """Tests that a curve whose value is constant over the entire range
    of the parameter (which is defined for all reals) has position at
    that constant which isn't the origin.
    """
    position = [1.61, -2.71, 3.14]

    def constant_curve_position(parameter: float) -> npt.ArrayLike:
        """Constant curve not at the origin."""
        return np.array(position)

    def constant_curve_first_derivative(parameter: float) -> npt.ArrayLike:
        """Derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    def constant_curve_second_derivative(parameter: float) -> npt.ArrayLike:
        """Second derivative of a constant curve."""
        return np.array([0.0, 0.0, 0.0])

    curve = AnalyticCurve(
        constant_curve_position,
        constant_curve_first_derivative,
        constant_curve_second_derivative,
    )

    for parameter in np.linspace(-10.0, 10.0, num=41):
        assert curve.position_at(parameter) == np.array(position), (
            "Fails to say that a constant curve not on the origin "
            "defined over all real parameters is equal to that constant "
            f"at {parameter}."
        )
