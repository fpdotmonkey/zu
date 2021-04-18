"""
Stuff for computing spacial curves based on analytically-defined
parametric functions.

Class AnalyticCurve:
    The object representing such a spacial curve.
"""

import logging
from typing import Callable, Optional, Sized

import numpy as np
import numpy.typing as npt


class AnalyticCurve:
    """A 1D curve in 3D space defined by analytical functions.  It
    provides information on the curve's derivatives, the curve's
    trihedral, and some other geometric information.

    There also are exceptions `BelowBounds` and `AboveBounds` which get
    raised when an out-of-bound parameter is given that contain
    information about the curve bounds.
    """

    class AboveBounds(Exception):
        """An exception raised when a value of the curve is requested at
        a parameter that is greater than the upper bound.
        """

        def __init__(self, message: str, upper_bound: float):
            """Generate an exception that contains a message and the
            upper bound that has been exceeded.
            """
            super().__init__(self)
            self._upper_bound = upper_bound

        @property
        def upper_bound(self):
            """The highest parameter this curve will admit before
            raising an exception.
            """
            return self._upper_bound

    class BelowBounds(Exception):
        """An exception raised when a value of the curve is requested at
        a parameter that is greater than the upper bound.
        """

        def __init__(self, message: str, lower_bound: float):
            """Pass in a friendly message to the user that they've
            exceeded their limit with respect to low-end parameters
            using message, and tell them what that limit was with
            lower_bound.
            """
            super().__init__(self)
            self._lower_bound = lower_bound

        @property
        def lower_bound(self):
            """The lowest legal parameter on this curve.  Lower will get
            you an exception.
            """
            return self._lower_bound

    def __init__(
        self,
        radius: Callable[[float], npt.ArrayLike],
        first_derivative: Callable[[float], npt.ArrayLike],
        second_derivative: Callable[[float], npt.ArrayLike],
        third_derivative: Callable[[float], npt.ArrayLike],
        bounds: Optional[Sized[Optional[float]]] = None,
    ):
        """Constructor for an analytic curve.  Note that there is no
        validation that the functions given to this constructor are
        correct.  It's the responsibility of users of this class to
        ensure whatever functions given are indeed the derivatives of
        the others.

        radius is a function that takes a real parameter and returns the
        displacement of each point on the curve from the origin.  It may
        also be a constant 3-vector if the curve is simply a point.

        first_derivative is the rate of change of the radius with
        respect to the parameter.  It can be given as a a function
        mapping a parameter to a 3-vector, a constant 3-vector if the
        first derivative is constant, or None if you're not interested
        in calculating anything based on the third derivative.

        second_derivative is the rate of the rate of change in the
        radius with respect to the parameter.  If the curve is
        understood to be a particle flying through space in time, the
        second derivative would be its acceleration.  What's passed in
        is the same as first_derivative.

        third_derivative is the thrice iterated derivative with respect
        to parameter of the radius.  If the curve is to be understood as
        economic price level (somehow in 3D), then former US President
        Richard Nixon would advertise that the third derivative is
        negative in saying "the rate of increase of inflation was
        decreasing" in hopes of deceiving you that the economy is good.
        The shape of this argument is just the same as for
        first_derivative and second_derivative.
        """
        np.seterr("raise")

        self._upper_bound: float = np.inf
        self._lower_bound: float = np.NINF
        if bounds:
            try:
                if len(bounds) != 2:
                    raise ValueError("`bounds` must be a length-2 Iterable.")
            except TypeError:
                raise ValueError("`bounds` must be a length-2 Iterable.")
            if bounds[0]:
                self._lower_bound = bounds[0]
            if bounds[1]:
                self._upper_bound = bounds[1]

        self._radius: Callable[[float], npt.ArrayLike] = radius
        self._first_derivative: Callable[
            [float], npt.ArrayLike
        ] = first_derivative
        self._second_derivative: Callable[
            [float], npt.ArrayLike
        ] = second_derivative
        self._third_derivative: Callable[
            [float], npt.ArrayLike
        ] = third_derivative

    def radius_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector coordinate at parameter and returns it."""
        self._check_bounds(parameter)
        return self._radius(parameter)

    def first_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector first derivative at parameter and
        returns it.
        """
        self._check_bounds(parameter)
        return self._first_derivative(parameter)

    def second_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector second derivative at parameter and
        returns it.
        """
        self._check_bounds(parameter)
        return self._second_derivative(parameter)

    def third_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector third derivative at parameter and
        returns it.
        """
        self._check_bounds(parameter)
        return self._third_derivative(parameter)

    def curvature_at(self, parameter: float) -> float:
        r"""This calculates the curvature of the curve at parameter.

        The curvature is the inverse of the radius of the circle tangent
        to the curve at parameter.  In general, the curvature is
        calculated as such

        .. math::
            :label: curvature

            k(t) = \frac{norm(r'(t) \times r''(t))}
                        {norm(r'(t))^3}
            (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.3))

        where k is curvature, t is parameter, r is the radius vector,
        and \times is the vector (cross) product.
        """
        first_derivative: npt.ArrayLike = self._first_derivative(parameter)
        second_derivative: npt.ArrayLike = self._second_derivative(parameter)
        logging.debug(
            "Calculating curvature of a curve with first derivative %s "
            "and second derivative %s.",
            first_derivative,
            second_derivative,
        )
        if np.array_equal(first_derivative, np.array([0.0, 0.0, 0.0])):
            logging.debug(
                "The first derivative equals [0, 0, 0], so curvature is "
                "0.0."
            )
            return 0.0

        curvature: float = (
            np.linalg.norm(
                np.cross(
                    first_derivative,
                    second_derivative,
                )
            )
            / (np.linalg.norm(first_derivative) ** 3)
        )
        logging.debug(
            "Calculating in the general case, the curvature is %s",
            curvature,
        )
        return curvature

    def torsion_at(self, parameter: float) -> float:
        r"""This calculates the curvature of the curve at parameter.

        The torsion of a curve is the rate of axial twist in that curve.
        It can be calculated in the general case with this equation.

        .. math::
            :label: torsion

            \chi(t) = \frac{(r'(t) \times r''(t)) \cdot r'''(t)}
                           {norm(r'(t) \times r''(t))^2}
            (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        where \chi is torsion, t is parameter, r is the radius vector,
        \times is the vector (cross) product, and \cdot is the scalar
        (dot) product.
        """
        first_derivative: npt.ArrayLike = self._first_derivative(parameter)
        second_derivative: npt.ArrayLike = self._second_derivative(parameter)
        third_derivative: npt.ArrayLike = self._third_derivative(parameter)
        logging.debug(
            "Calculating torsion of a curve with first derivative %s, "
            "second derivative %s, and third derivative %s.",
            first_derivative,
            second_derivative,
            third_derivative,
        )

        if np.isclose(
            np.linalg.norm(np.cross(first_derivative, second_derivative)), 0.0
        ):
            logging.debug(
                "The first and second derivatives are parallel, so the "
                "torsion must be 0."
            )
            return 0.0

        torsion: float = np.dot(
            np.cross(first_derivative, second_derivative), third_derivative
        ) / (
            np.linalg.norm(np.cross(first_derivative, second_derivative)) ** 2
        )
        logging.debug(
            "Calculating in the general case, the torsion is %s",
            torsion,
        )
        return torsion

    def tangent_vector_at(self, parameter: float) -> npt.ArrayLike:
        r"""This calculates the tangent vector of the curve at
        parameter.

        A curve's tangent vector is parallel with the first derivative
        and lies along intersection of the osculating plane and the
        rectifying plane.  It can be calculated as the following.

        .. math::
            :label: tangent

            T(t) = normalize(r'(t))
            (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.2))

        where T is the tangent vector, t is parameter, and r is the
        radius vector.
        """
        first_derivative: npt.ArrayLike = self._first_derivative(parameter)
        logging.debug(
            "Calculating the tangent vector of a curve with first "
            "derivative %s.",
            first_derivative,
        )

        if np.isclose(np.linalg.norm(first_derivative), 0.0):
            logging.debug(
                "The first derivative equals [0, 0, 0], so the tangent "
                "vector is [0, 0, 0]."
            )
            return np.array([0.0, 0.0, 0.0])

        tangent_vector: npt.ArrayLike = first_derivative / np.linalg.norm(
            first_derivative
        )
        logging.debug(
            "Calculating in the general case, the tangent vector is %s.",
            tangent_vector,
        )
        return tangent_vector

    def normal_vector_at(self, parameter: float) -> npt.ArrayLike:
        r"""This calculates the normal vector of the curve at
        parameter.

        The normal vector of a curve, also called the principal normal
        of the curve, lies parallel to the second derivative of the
        curve in its natural parameterization.  It also is found at the
        intersection of the osculating and normal planes.  It is
        strictly perpendicular to the tangent vector.  Analytically, it
        can be computed as this.

        .. math::
            :label: normal

            n(t) = (\frac{r''(t)}
                         {norm(r'(t))^2}
                    - r'(t) \frac{r'(t) \cdot r''(t)}
                                 {norm(r'(t))^4}
                   ) / k(t)
            (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        where n is the normal vector, t is parameter, r is the radius
        vector, k is the curvature at t, and \cdot is the scalar (dot)
        product.

        In cases where the normal vector would be ill-defined, i.e. when
        the first and second derivatives of the curve are parallel (in
        other words, a straight line) or when the first derivative is
        the zero vector, the normal vector will be made to be the zero
        vector.
        """
        first_derivative: npt.ArrayLike = self._first_derivative(parameter)
        second_derivative: npt.ArrayLike = self._second_derivative(parameter)
        logging.debug(
            "Calculating the normal vector of a curve with first "
            "derivative %s and second derivative %s.",
            first_derivative,
            second_derivative,
        )

        if np.isclose(
            np.linalg.norm(np.cross(first_derivative, second_derivative)), 0.0
        ):
            logging.debug(
                "The first and second derivatives are parallel, so the "
                "normal vector must be [0, 0, 0]."
            )
            return [0.0, 0.0, 0.0]

        if np.isclose(np.linalg.norm(first_derivative), 0.0):
            logging.debug(
                "The first derivative equals [0, 0, 0], so the normal "
                "vector is [0, 0, 0]."
            )
            return np.array([0.0, 0.0, 0.0])

        normal_vector: npt.ArrayLike = (
            np.power(np.linalg.norm(first_derivative), 2) * second_derivative
            - np.dot(first_derivative, second_derivative) * first_derivative
        ) / (
            np.linalg.norm(np.cross(first_derivative, second_derivative))
            * np.linalg.norm(first_derivative)
        )
        logging.debug(
            "Calculating in the general case, the normal vector is %s.",
            normal_vector,
        )
        return normal_vector

    def binormal_vector_at(self, parameter: float) -> npt.ArrayLike:
        r"""This calculates the binormal vector of the curve at
        parameter.

        The binormal vector of a curve is the vector along the
        intersection of the normal and rectifying planes and that also
        forms a right-handed triple with the tangent and normal vectors.
        Generally, it can be calculated with this equation.

        .. math::
            :label: binormal

            b(t) = \frac{r'(t) \times r''(t)}
                        {k(t) norm(r'(t))}
            (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        where b is torsion, t is parameter, r is the radius vector, k is
        the curvature at t, and \times is the vector (cross) product.
        """
        first_derivative: npt.ArrayLike = self._first_derivative(parameter)
        second_derivative: npt.ArrayLike = self._second_derivative(parameter)
        logging.debug(
            "Calculating the binormal vector of a curve with first "
            "derivative %s and second derivative %s.",
            first_derivative,
            second_derivative,
        )

        if np.isclose(
            np.linalg.norm(np.cross(first_derivative, second_derivative)), 0.0
        ):
            logging.debug(
                "The first and second derivatives are parallel, so the "
                "binormal vector must be [0, 0, 0]."
            )
            return [0.0, 0.0, 0.0]

        binormal_vector: npt.ArrayLike = np.cross(
            first_derivative, second_derivative
        ) / np.linalg.norm(np.cross(first_derivative, second_derivative))
        logging.debug(
            "Calculating in the general case, the binormal vector is %s.",
            binormal_vector,
        )
        return binormal_vector

    def _check_bounds(self, parameter: float) -> None:
        """This compares the parameter to the upper and lower bounds,
        and if it's out of bounds, it raises the appropriate exception.
        """
        if parameter > self._upper_bound:
            raise self.AboveBounds(
                f"Parameter {parameter} is greater than the upper bound "
                f"parameter {self._upper_bound}.",
                self._upper_bound,
            )
        if parameter < self._lower_bound:
            raise self.BelowBounds(
                f"Parameter {parameter} is less than the lower bound "
                f"parameter {self._lower_bound}.",
                self._lower_bound,
            )
