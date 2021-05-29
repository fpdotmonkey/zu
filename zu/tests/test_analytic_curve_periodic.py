"""
Tests for curves where the beginning and end points are the same.
"""

import logging

import numpy as np
import pytest

from zu.analytic_curve import AnalyticCurve


npt = np.typing

logging.getLogger().setLevel(logging.DEBUG)


def test_periodic_curve_with_no_defined_bounds() -> None:
    """If a curve is made to be periodic, but the bounds aren't defined,
    a ValueError should be raised.
    """
    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            periodic=True,
        )


def test_periodic_curve_with_infinite_bounds() -> None:
    """If a curve is made to be periodic, but the bounds include
    infinity, a ValueError should be raised.
    """
    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(np.NINF, np.inf),
            periodic=True,
        )

    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(np.NINF, 0),
            periodic=True,
        )

    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(0, np.inf),
            periodic=True,
        )


def test_aperiodic_curve() -> None:
    """A curve with `periodic=False` should behave like normal."""
    AnalyticCurve(
        (
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
        ),
        bounds=(np.NINF, np.inf),
        periodic=False,
    )


def test_periodic_line() -> None:
    """If a curve is periodic, then if you plug in out-of-bounds
    parameters, then they will lead to in-bounds calculations.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
        ),
        bounds=(0.0, 1.0),
        periodic=True,
    )

    for parameter in np.linspace(-10, 10, num=21):
        np.testing.assert_allclose(
            curve.radius_at(parameter),
            np.mod(parameter, 1.0) * np.array(velocity),
            err_msg="Fails to calculate on a periodic line.",
        )


def test_verify_cyclic_closed() -> None:
    """If the user says that the curve is cyclic closed, then there
    should be a step to verify that the curve is actually closed.
    """
    with pytest.raises(ValueError):
        # a straight line is not cyclic closed in uncurved space
        velocity = [1, 0, 0]
        AnalyticCurve(
            (
                lambda parameter: parameter * np.array(velocity),
                lambda parameter: np.array(velocity),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(0.0, 1.0),
            periodic=True,
            cyclic_closed=True,
        )

    # one period of a circle is periodic closed
    AnalyticCurve(
        (
            lambda parameter: np.array(
                [np.cos(parameter), np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [-np.sin(parameter), np.cos(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [-np.cos(parameter), -np.sin(parameter), 0.0]
            ),
            lambda parameter: np.array(
                [np.sin(parameter), -np.cos(parameter), 0.0]
            ),
        ),
        bounds=(0.0, 2.0 * np.pi),
        periodic=True,
        cyclic_closed=True,
    )


def test_cyclic_closed_only_if_periodic() -> None:
    """If `cyclic_closed=True`, but `periodic=False`, then a ValueError
    should be raised.
    """
    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array(
                    [np.cos(parameter), np.sin(parameter), 0.0]
                ),
                lambda parameter: np.array(
                    [-np.sin(parameter), np.cos(parameter), 0.0]
                ),
                lambda parameter: np.array(
                    [-np.cos(parameter), -np.sin(parameter), 0.0]
                ),
                lambda parameter: np.array(
                    [np.sin(parameter), -np.cos(parameter), 0.0]
                ),
            ),
            bounds=(0.0, 2.0 * np.pi),
            periodic=False,
            cyclic_closed=True,
        )
