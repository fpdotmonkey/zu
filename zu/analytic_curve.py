"""
Stuff for computing spacial curves based on analytically-defined
parametric functions.

Class AnalyticCurve:
    The object representing such a spacial curve.
"""

import logging
from typing import Callable, Tuple

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

            :param      message:      A message signaling above-bounds.
            :type       message:      str
            :param      upper_bound:  The curve's upper bound parameter.
            :type       upper_bound:  float
            """
            super().__init__(self)
            self._message = message
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

            :param      message:      A message signaling below-bounds.
            :type       message:      str
            :param      lower_bound:  The curve's lower bound parameter.
            :type       lower_bound:  float
            """
            super().__init__(self)
            self._message = message
            self._lower_bound = lower_bound

        @property
        def lower_bound(self):
            """The lowest legal parameter on this curve.  Lower will get
            you an exception.
            """
            return self._lower_bound

    def __init__(
        self,
        coordinates: Tuple[
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
        ],
        bounds: Tuple[float, float] = (np.NINF, np.inf),
        periodic: bool = False,
        cyclic_closed: bool = False,
    ):
        """Constructor for an analytic curve.  Note that there is no
        validation that the functions given to this constructor are
        correct.  It's the responsibility of users of this class to
        ensure whatever functions given are indeed the derivatives of
        the others.

        :param      coordinates:    A tuple of functions heads.  Each
                                    function must take a float and
                                    return a numpy array-like.  The
                                    zeroth function should represent the
                                    "radius" of the curve (i.e. the
                                    coordinates that the curve passes
                                    through) and each function at index
                                    i beyond that represents the ith
                                    derivative.  You must include all
                                    derivatives (which are described
                                    below) through the 3rd.
        :type       coordinates:    Tuple[
                                        Callable[[float], npt.ArrayLike],
                                        Callable[[float], npt.ArrayLike],
                                        Callable[[float], npt.ArrayLike],
                                        Callable[[float], npt.ArrayLike],
                                    ]
        :param      bounds:         The (upper_limit, lower_limit) for a
                                    finite curve.  These limits can also
                                    be +/- infinity if you want the
                                    curve to extend to infinity only in
                                    one direction for example.  If
                                    bounds does not have length of 2,
                                    then a ValueError will be raised, as
                                    well as if the former bound is
                                    greater than the latter bound.
        :type       bounds:         Tuple[float, float]
        :param      periodic:       This flags that the curve is
                                    periodic (i.e. there exists some
                                    number p > 0 where
                                    `radius_at(parameter +/- p * k)
                                    == radius_at(parameter)`, where k is
                                    an integer).  If periodic is true,
                                    then infinity may not be one of the
                                    bounds, else a ValueError will be
                                    raised.
        :type       periodic:       bool
        :param      cyclic_closed:  This should be set true if you want
                                    to verify that not only is your
                                    curve periodic, but that it is
                                    continuous (C0) across the bounds
                                    seems.  If it is not, then a
                                    ValueError will be raised.  Also, if
                                    cyclic_closed is True and periodic
                                    is False, then a ValueError will be
                                    raised.
        :type       cyclic_closed:  bool

        :raises     ValueError: If there aren't enough coordinates to
                                get the third derivative.
        :raises     ValueError: If `bounds` doesnt have length of two.
        :raises     ValueError: If the lower bound is greater than the
                                upper.
        :raises     ValueError: If the curve is periodic and one of the
                                bounds is infinite.
        :raises     ValueError: If `cyclic_closed` and not `periodic`.
        :raises     ValueError: If the radius function evaluated at
                                bounds[0] is not equal to the radius
                                function evaluated at bounds[1].

        Here's more information on coordinates.

        The radius  takes a real parameter and returns the displacement
        of each point on the curve from the origin.

        The first derivative is the rate of change of the radius with
        respect to the parameter.  It must be given as a a function
        mapping a parameter to a 3-vector.

        The second derivative is the rate of the rate of change in the
        radius with respect to the parameter.  If the curve is
        understood to be a particle flying through space in time, the
        second derivative would be its acceleration.  What's passed in
        is the same as first_derivative.

        The third derivative is the thrice iterated derivative with
        respect to parameter of the radius.  If the curve is to be
        understood as economic price level (somehow in 3D), then former
        US President Richard Nixon would advertise that the third
        derivative is negative in saying "the rate of increase of
        inflation was decreasing" in hopes of deceiving you that the
        economy is good.  The shape of this argument is just the same as
        for the first and second derivatives.
        """
        np.seterr("raise")

        if len(coordinates) < 4:
            raise ValueError(
                "coordinates array must include the radius, first "
                "derivative, second derivative, and third derivative."
            )
        if len(bounds) != 2:
            raise ValueError("`bounds` must be a length-2 tuple.")
        if bounds[0] > bounds[1]:
            raise ValueError(
                "bounds[1] must be greater than or equal to bounds[0]."
            )
        if periodic and (np.inf in bounds or np.NINF in bounds):
            raise ValueError(
                "Periodic curves may not take infinite parameters."
            )
        if cyclic_closed and not periodic:
            raise ValueError(
                "cyclic_closed is only valid if periodic is True."
            )

        self._coordinates: Tuple[
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
            Callable[[float], npt.ArrayLike],
        ] = coordinates

        self._lower_bound: float = bounds[0]
        self._upper_bound: float = bounds[1]

        if (
            cyclic_closed
            and not np.isclose(
                self._coordinates[0](self._lower_bound),
                self._coordinates[0](self._upper_bound),
            ).all()
        ):
            raise ValueError(
                "This curve is not cyclic closed, but you "
                "said it should be."
            )

        self._periodic: bool = periodic
        self._cyclic_closed: bool = cyclic_closed

    def radius_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector coordinate at parameter and returns it.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The radius vector.
        :rtype:     numpy.typing.ArrayLike
        """
        radius = self._coordinates[0](self._bounded(parameter))
        logging.debug(
            "Calculated radius of %s from parameter %f.", radius, parameter
        )
        return radius

    def first_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector first derivative at parameter and
        returns it.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The first derivative vector.
        :rtype:     numpy.typing.ArrayLike
        """
        first_derivative = self._coordinates[1](self._bounded(parameter))
        logging.debug(
            "Calculated first derivative of %s from parameter %f.",
            first_derivative,
            parameter,
        )
        return first_derivative

    def second_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector second derivative at parameter and
        returns it.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The second derivative vector.
        :rtype:     numpy.typing.ArrayLike
        """
        second_derivative = self._coordinates[2](self._bounded(parameter))
        logging.debug(
            "Calculated second derivative of %s from parameter %f.",
            second_derivative,
            parameter,
        )
        return second_derivative

    def third_derivative_at(self, parameter: float) -> npt.ArrayLike:
        """Computes the 3-vector third derivative at parameter and
        returns it.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The third derivative vector.
        :rtype:     numpy.typing.ArrayLike
        """
        third_derivative = self._coordinates[3](self._bounded(parameter))
        logging.debug(
            "Calculated third derivative of %s from parameter %f.",
            third_derivative,
            parameter,
        )
        return third_derivative

    def curvature_at(self, parameter: float) -> float:
        r"""This calculates the curvature of the curve at parameter.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The curvature.
        :rtype:     float

        The curvature is the inverse of the radius of the circle tangent
        to the curve at parameter.  In general, the curvature is
        calculated as such

        :f[
            :label: curvature

            k(t) = \frac{norm(r'(t) \times r''(t))}
                        {norm(r'(t))^3}
        :f]
        (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.3))

        Where k is curvature, t is parameter, r is the radius vector,
        and \times is the vector (cross) product.
        """
        first_derivative: npt.ArrayLike = self.first_derivative_at(parameter)
        second_derivative: npt.ArrayLike = self.second_derivative_at(parameter)
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

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The torsion of the curve.
        :rtype:     float

        The torsion of a curve is the rate of axial twist in that curve.
        It can be calculated in the general case with this equation.

        :f[
            :label: torsion

            \chi(t) = \frac{(r'(t) \times r''(t)) \cdot r'''(t)}
                           {norm(r'(t) \times r''(t))^2}
        :f]
        (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        Where :f$ \chi :f$ is torsion, :f$ t :f$ is parameter, :f$ r :f$
        is the radius vector, :f$ \times :f$ is the vector (cross)
        product, and :f$ \cdot :f$ is the scalar (dot) product.
        """
        first_derivative: npt.ArrayLike = self.first_derivative_at(parameter)
        second_derivative: npt.ArrayLike = self.second_derivative_at(parameter)
        third_derivative: npt.ArrayLike = self.third_derivative_at(parameter)
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

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The tangent vector of the curve.
        :rtype:     numpy.typing.ArrayLike

        A curve's tangent vector is parallel with the first derivative
        and lies along intersection of the osculating plane and the
        rectifying plane.  It can be calculated as the following.

        :f[
            T(t) = normalize(r'(t))
        :f]
        (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.2))

        Where :f$ T :f$ is the tangent vector, :f$ t :f$ is parameter,
        and :f$ r :f$ is the radius vector.
        """
        first_derivative: npt.ArrayLike = self.first_derivative_at(parameter)
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

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The normal vector of the curve.
        :rtype:     numpy.typing.ArrayLike

        The normal vector of a curve, also called the principal normal
        of the curve, lies parallel to the second derivative of the
        curve in its natural parameterization.  It also is found at the
        intersection of the osculating and normal planes.  It is
        strictly perpendicular to the tangent vector.  Analytically, it
        can be computed as this.

        :f[
            :label: normal

            n(t) = (\frac{r''(t)}
                         {norm(r'(t))^2}
                    - r'(t) \frac{r'(t) \cdot r''(t)}
                                 {norm(r'(t))^4}
                   ) / k(t)
        :f]
        (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        Where :f$ n :f$ is the normal vector, :f$ t :f$ is parameter,
        :f$ r :f$ is the radius vector, :f$ k :f$ is the curvature at
        :f$ t :f$, and :f$ \cdot :f$ is the scalar (dot) product.

        In cases where the normal vector would be ill-defined, i.e. when
        the first and second derivatives of the curve are parallel (in
        other words, a straight line) or when the first derivative is
        the zero vector, the normal vector will be made to be the zero
        vector.
        """
        first_derivative: npt.ArrayLike = self.first_derivative_at(parameter)
        second_derivative: npt.ArrayLike = self.second_derivative_at(parameter)
        logging.debug(
            "Calculating the normal vector of a curve with first "
            "derivative %s and second derivative %s.",
            first_derivative,
            second_derivative,
        )

        if np.isclose(np.linalg.norm(first_derivative), 0.0):
            logging.debug(
                "The first derivative equals [0, 0, 0], so the normal "
                "vector is [0, 0, 0]."
            )
            return np.array([0.0, 0.0, 0.0])

        if np.isclose(
            np.linalg.norm(np.cross(first_derivative, second_derivative)), 0.0
        ):
            logging.debug(
                "The first and second derivatives are parallel, so the "
                "normal vector must be [0, 0, 0]."
            )
            return [0.0, 0.0, 0.0]

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

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   The binormal vector of the curve.
        :rtype:     numpy.typing.ArrayLike

        The binormal vector of a curve is the vector along the
        intersection of the normal and rectifying planes and that also
        forms a right-handed triple with the tangent and normal vectors.
        Generally, it can be calculated with this equation.

        :f[
            :label: binormal

            b(t) = \frac{r'(t) \times r''(t)}
                        {k(t) norm(r'(t))}
        :f]
        (Nikolai Golovanov, "Geometric Modeling", eq. (1.1.4))

        Where :f$ b :f$ is torsion, :f$ t :f$ is parameter, :f$ r :f$ is
        the radius vector, :f$ k :f$ is the curvature at :f$ t :f$, and
        :f$ \times :f$ is the vector (cross) product.

        In cases where the binormal vector would be ill-defined, i.e.
        when the first and second derivatives of the curve are parallel
        (in other words, a straight line) or when the first derivative
        is the zero vector, the normal vector will be made to be the
        zero vector.
        """
        first_derivative: npt.ArrayLike = self.first_derivative_at(parameter)
        second_derivative: npt.ArrayLike = self.second_derivative_at(parameter)
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

    def _bounded(self, parameter: float) -> float:
        """This compares the parameter to the upper and lower bounds,
        and if it's out of bounds, it either wraps it into bound if the
        curve is periodic or raises the appropriate exception if its
        not.

        :param      parameter:  The parameter along the curve.
        :type       parameter:  float

        :returns:   An in-bounds parameter.
        :rtype:     float

        :raises     AboveBounds: If a too-high parameter is given in a
                                 non-periodic curve.
        :raises     BelowBounds: If a too-low parameter is given in a
                                 non-periodic curve
        """
        if self._periodic:
            modded_parameter = (
                np.mod(parameter, self._upper_bound - self._lower_bound)
                + self._lower_bound
            )
            logging.debug(
                "Parameter %f is out of bounds on a periodic curve.  "
                "Moving it to %f.",
                parameter,
                modded_parameter,
            )
            return modded_parameter

        if self._lower_bound <= parameter <= self._upper_bound:
            return parameter

        if parameter > self._upper_bound:
            logging.debug(
                "Parameter %f is greater than the maximum parameter "
                "%f.  Raising AboveBounds.",
                parameter,
                self._upper_bound,
            )
            raise self.AboveBounds(
                f"Parameter {parameter} is greater than the upper bound "
                f"parameter {self._upper_bound}.",
                self._upper_bound,
            )
        logging.debug(
            "Parameter %f is less than the minimum parameter "
            "%f.  Raising BelowBounds.",
            parameter,
            self._lower_bound,
        )
        raise self.BelowBounds(
            f"Parameter {parameter} is less than the lower bound "
            f"parameter {self._lower_bound}.",
            self._lower_bound,
        )
