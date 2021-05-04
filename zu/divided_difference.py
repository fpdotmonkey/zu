"""
An implementation of the divided difference algorithm.

Function divided_difference:
    Given a scalar function and list of knots, then it give the output of the
    divided difference algorithm; useful for such applications as
    interpolation.
"""

import logging
from typing import Callable

import numpy as np


def divided_difference(
    function: Callable[[float], float], knots: np.ndarray
) -> float:
    r"""Computes the divided difference algorithm over `knots` with the
    function `function`.  It's useful for things like interpolation.

    :param      function:    The function over which you'd like to take
                             the divided difference
    :type       function:    Callable[[float], float]
    :param      knots:       An array of knots with which to compute the
                             divided difference
    :type       knots:       np.ndarray

    :returns:   The value of the divided difference with the knots over
                the function
    :rtype:     float

    :raises     ValueError:  If the knots array has length 0.

    Given a single knot, the divided difference is simply the function
    evaluated at that knot.

    :f[
        divided_difference(f, [0]) = f(0)
    :f]
    (Nikolai Golovanov, "Geometric Modeling", pg. 43)

    When there are multiple knots, the divided difference can be
    computed recursively with subsets of the knots.  Here first is the
    two-knot case.

    :f[
        divided_difference(f, [0, 1]) = \frac{f(1) - f(0)}
                                             {1 - 0}
    :f]
    (Nikolai Golovanov, "Geometric Modeling", eq. (1.7.1))

    And here is the general m-knot case.

    :f[
        divided_differenct(f, [t_0, ..., t_m])
            = \frac{divided_difference(f, [t_1, ..., t_m])
                    - divided_difference(f, [t_0, ..., t_{m-1}])}
                   {t_m - t_0}
    :f]
    (Nikolai Golovanov, "Geometric Modeling", eq. (1.7.2))

    Where t_i is the ith knot

    Perhaps you can see how this could be used for higher and higher
    orders of interpolation.  In particular, this can be used to
    generate coefficients of a truncated taylor series.

    :f[
        taylor_series(f, t) =
            divided_difference(f, [t0])
            + divided_difference(f, [t0, t1])(t - t0)
            + ...
            + divided_difference(f, [t0, t1, ..., tm])
              * (t - t0) * (t - t1) * ... * (t - tm)
    :f]
    (Nikolai Golovanov, "Geometric Modeling", eq. (1.7.5))

    Here though, I use an analytical method which is harder to reason
    about but is nicer to compute.

    :f[
        divided_difference(f, [t_0, ..., t_m]) =
            \sum_{j=0}^m \frac{f(t_j)}
                              {W_{0,m}'(t_j)}
        W_{0,m}'(t_i) =
            (t_i - t_0) * ...
            * (t_i - t_{i-1}) * (t_i - t_{i+1}) * ...
            * (t_i - t_m)
    :f]
    (Nikolai Golovanov, "Geometric Modeling", eq. (1.7.3))
    """
    if knots.shape[0] == 0:
        raise ValueError("knots must be a list of floats with length >= 1")
    if knots.shape[0] == 1:
        difference = function(knots[0])
        logging.debug(
            "Calculating in the case of only one knot, the divided "
            "difference is %e",
            difference,
        )
        return difference

    difference = float(
        np.sum(
            np.array(
                [
                    function(knots[i])
                    / _divided_difference_denominator(i, knots)
                    for i in range(knots.shape[0])
                ]
            )
        )
    )
    logging.debug(
        "Calculating in the general case, the divided difference is %e.",
        difference,
    )
    return difference


def _divided_difference_denominator(
    knot_index: int, knots: np.ndarray
) -> float:
    """
    Computes the denominator of an analytically-calculated divided
    difference.

    :param      knot_index:  The index of the particular knot the
                             denominator should be calculated for.
    :type       knot_index:  int
    :param      knots:       The array of knots.
    :type       knots:       numpy.ndarray

    :returns:   The denominator of the divided difference.
    :rtype:     float
    """
    knot = knots[knot_index]
    return float(
        np.prod(
            np.array(
                [
                    knot - knots[i]
                    for i in range(knots.shape[0])
                    if i != knot_index
                ]
            )
        )
    )
