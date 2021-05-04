"""
Tests to validate that the divided_difference function works alright.
"""

import numpy as np
import pytest

from zu.divided_difference import divided_difference


def test_empty_knot_list_raises_value_error():
    """Should an empty list be passed in for `knots`, then you should
    get a ValueError.
    """
    with pytest.raises(ValueError):
        divided_difference(np.sin, np.array([]))


def test_single_knot_gives_function_at_knot():
    """If there's a single knot, then it should return the passed-in
    function evaluated at that knot.
    """
    for knot in np.linspace(-10, 10, num=41):
        assert divided_difference(np.sin, np.array([knot])) == np.sin(knot), (
            "Fails to give the divided difference with a single knot as "
            "the function evaluated at that knot."
        )


def test_two_knot_divided_difference():
    """The divided difference should evaulate correctly when there's two
    knots.
    """
    assert divided_difference(np.sin, np.array([1, 10])) == (
        np.sin(10) - np.sin(1)
    ) / (10 - 1), (
        "Fails to correctly evaluate the divided difference with a two "
        "knots."
    )


def test_many_knots_divided_difference():
    """If there are many knots, then the divided difference should still
    work.
    """
    assert np.isclose(
        divided_difference(np.sin, np.array(list(range(20)))),
        -(8 * np.cos(19 / 2) * np.sin(1 / 2) ** 19) / 1_856_156_927_625
        # approximately 3.68933*10^-18
    ), (
        "Fails to correctly evaluate the divided difference when there "
        "are many knots."
    )
