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
    assert np.isclose(
        divided_difference(np.sin, np.array([1, 10])),
        (np.sin(10) - np.sin(1)) / (10 - 1),
    ), (
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


def test_m_knot_divided_difference_of_m_degree_polynomial_is_const():
    """If you take the divided difference with m knots of a polynomial
    of degree m, then the result should be the coefficient of the
    highest-order term.  This per (Nikolai Golovanov, "Geometric
    Modeling", eq. (1.7.6))
    """
    assert np.isclose(
        divided_difference(
            lambda x: 17 * x ** 3 + x ** 2 + x + 1, np.array([0, 1, 2, 3])
        ),
        17,
    ), (
        "Fails to give the leading coefficient of the polynomial as the "
        "divided difference."
    )


def test_m_1_knot_divided_difference_of_m_degree_polynomial_is_zero():
    """Taking a divided difference with m + 1 knots of a polynomial
    function of degree m should be 0.
    """
    assert np.isclose(
        divided_difference(
            lambda x: 17 * x ** 3 + x ** 2 + x + 1, np.array([0, 1, 2, 3, 4])
        ),
        0,
    ), "Fails to give zero of the polynomial as the divided difference."


def test_symmetric_with_knot_order():
    """The value of the divided difference should not depend on the
    order of the knots.
    """
    assert np.isclose(
        divided_difference(np.sin, np.array([1, 10, 2, 9, 3, 8, 4, 7, 5, 6])),
        divided_difference(np.sin, np.array([10, 8, 6, 4, 2, 1, 3, 5, 7, 9])),
    ), "Fails to give that the knot order doesn't matter."


def test_is_linear():
    """If a function is a linear combination of other functions, then
    its divided difference is the linear combination of the those
    constituent function's divided differences.
    """
    assert np.isclose(
        divided_difference(
            lambda x: 4 * np.sin(x) + 2 * np.cos(x), np.array([0, 1, 2])
        ),
        4 * divided_difference(np.sin, np.array([0, 1, 2]))
        + 2 * divided_difference(np.cos, np.array([0, 1, 2])),
    ), "Fails to make divided difference linear."


def test_leibniz_formula():
    """If a function is the product of two functions, then the divided
    difference of this function should be the sum of each product of the
    divided differences of the constituent functions where one function
    takes knots 0 to i and the other takes knots i to m, where m is the
    total number of knots.  This per (Nikolai Golovanov, "Geometric
    Modeling", eq. (1.7.7))
    """
    knots = np.array(list(range(10)))
    assert np.isclose(
        divided_difference(lambda x: np.sin(x) * np.cos(x), knots),
        np.sum(
            [
                divided_difference(np.sin, knots[: i + 1])
                * divided_difference(np.cos, knots[i:])
                for i in range(len(knots))
            ]
        ),
    ), "Fails to satisfy the Leibniz formula."
