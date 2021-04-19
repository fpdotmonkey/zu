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


def test_infinite_bounds_inifinite_curve() -> None:
    """If the keyword argument `bounds=(-inf, inf)` is given, then the
    curve should extend off to +/- infinity, as normal.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(np.NINF, np.inf),
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

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(np.NINF, 10.0),
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
    assert np.isclose(
        above_bounds.value.upper_bound, 10.0
    ), "Fails to send the upper bound with the AboveBounds exception."


def test_curve_with_lower_bound() -> None:
    """A curve where the keyword argument `upper_bound=[float]` is
    supplied should result in a curve where the input parameter can be
    in the interval [lower_bound, inf).
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, np.inf),
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
    assert np.isclose(
        below_bounds.value.lower_bound, -10.0
    ), "Fails to send the lower bound with the BelowBounds exception."


def test_curve_with_upper_and_lower_bounds() -> None:
    """A curve with finite upper and lower bounds should be valid
    between those bounds (inclusively) and should raise an AboveBounds
    exception when greater than the valid interval and a BelowBounds
    exception when less than the valid interval.
    """
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
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
    assert np.isclose(
        below_bounds.value.lower_bound, -10.0
    ), "Fails to send the lower bound with the BelowBounds exception."

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.radius_at(10.0001)
    assert np.isclose(
        above_bounds.value.upper_bound, 10.0
    ), "Fails to send the upper bound with the AboveBounds exception."


def test_misordered_bounds() -> None:
    """The second bound should be greater than the first.  If it's not,
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
            bounds=(10.0, -10.0),
        )


def test_equal_bounds() -> None:
    """If the bounds are equal, then they should return valid results at
    that points where the bounds are equal.
    """
    curve = AnalyticCurve(
        (
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
            lambda parameter: np.array([0, 0, 0]),
        ),
        bounds=(10.0, 10.0),
    )

    np.testing.assert_allclose(
        curve.radius_at(10.0),
        np.array([0, 0, 0]),
        err_msg="Fails to allow equal bounds.",
    )


def test_wrong_length_bounds() -> None:
    """If you give `bounds` as something that's not length-2, it should
    raise a ValueError.
    """
    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(-10.0, 10.0, 11.0),  # type: ignore
        )

    with pytest.raises(ValueError):
        AnalyticCurve(
            (
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
                lambda parameter: np.array([0, 0, 0]),
            ),
            bounds=(-10.0,),  # type: ignore
        )


def test_finite_first_derivative() -> None:
    """Finite curves should nominally word for first derivatives."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.first_derivative_at(parameter),
            np.array(velocity),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "first derivative.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.first_derivative_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the first derivative."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.first_derivative_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the first derivative."
    )


def test_finite_second_derivative() -> None:
    """Finite curves should nominally word for second derivatives."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.second_derivative_at(parameter),
            np.array([0, 0, 0]),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "second derivative.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.second_derivative_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the second derivative."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.second_derivative_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the second derivative."
    )


def test_finite_third_derivative() -> None:
    """Finite curves should nominally word for third derivatives."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.third_derivative_at(parameter),
            np.array([0, 0, 0]),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "third derivative.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.third_derivative_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the third derivative."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.third_derivative_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the third derivative."
    )


def test_finite_tangent_vector() -> None:
    """Finite curves should nominally word for tangent vectors."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.tangent_vector_at(parameter),
            np.array(velocity),
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "tangent vector.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.tangent_vector_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the tangent vector."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.tangent_vector_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the tangent vector."
    )


def test_finite_normal_vector() -> None:
    """Finite curves should nominally word for normal vectors."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        assert np.isclose(
            np.dot(
                curve.normal_vector_at(parameter),
                curve.tangent_vector_at(parameter),
            ),
            0.0,
        ), (
            "Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "normal vector."
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.normal_vector_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the normal vector."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.normal_vector_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the normal vector."
    )


def test_finite_binormal_vector() -> None:
    """Finite curves should nominally word for binormal vectors."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        assert np.isclose(
            np.dot(
                curve.binormal_vector_at(parameter),
                curve.tangent_vector_at(parameter),
            ),
            0.0,
        ) and np.isclose(
            np.dot(
                curve.binormal_vector_at(parameter),
                curve.normal_vector_at(parameter),
            ),
            0.0,
        ), (
            "Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "binormal vector."
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.binormal_vector_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the binormal vector."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.binormal_vector_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the binormal vector."
    )


def test_finite_curvature() -> None:
    """Finite curves should nominally word for curvatures."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.curvature_at(parameter),
            0.0,
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "curvature.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.curvature_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the curvature."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.curvature_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the curvature."
    )


def test_finite_torsion() -> None:
    """Finite curves should nominally word for torsions."""
    velocity = [1.0, 0.0, 0.0]

    curve = AnalyticCurve(
        (
            lambda parameter: parameter * np.array(velocity),
            lambda parameter: np.array(velocity),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
            lambda parameter: np.array([0.0, 0.0, 0.0]),
        ),
        bounds=(-10.0, 10.0),
    )

    for parameter in np.linspace(-10.0, 10, num=21):
        np.testing.assert_allclose(
            curve.torsion_at(parameter),
            0.0,
            err_msg="Fails to generate a curve where parameters in the "
            "interval [lower_bound, upper_bound] are valid on the "
            "torsion.",
        )

    with pytest.raises(AnalyticCurve.BelowBounds) as below_bounds:
        curve.torsion_at(-10.0001)
    assert np.isclose(below_bounds.value.lower_bound, -10.0), (
        "Fails to send the lower bound with the BelowBounds exception on "
        "the torsion."
    )

    with pytest.raises(AnalyticCurve.AboveBounds) as above_bounds:
        curve.torsion_at(10.0001)
    assert np.isclose(above_bounds.value.upper_bound, 10.0), (
        "Fails to send the upper bound with the AboveBounds exception on "
        "the torsion."
    )
