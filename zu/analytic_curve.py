"""
Stuff for computing spacial curves based on analytically-defined
parametric functions.

Class AnalyticCurve:
    The object representing such a spacial curve.
"""

from numbers import Number
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt


class AnalyticCurve:
    """A 1D curve in 3D space defined by analytical functions.  It
    provides information on the curve's derivatives, the curve's
    trihedral, and some other geometric information.
    """

    def __init__(
        self,
        radius: Union[Callable[[Number], npt.ArrayLike], npt.ArrayLike],
        first_derivative: Optional[
            Union[Callable[[Number], npt.ArrayLike], npt.ArrayLike]
        ],
        second_derivative: Optional[
            Union[Callable[[Number], npt.ArrayLike], npt.ArrayLike]
        ],
        third_derivative: Optional[
            Union[Callable[[Number], npt.ArrayLike], npt.ArrayLike]
        ],
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
        in calculating anything based on the third derivative.  It's
        required to calculate the tangent, normal, and binormal vectors,
        and the curvature and torsion of the curve.  Additionally, if
        this is None, then all higher-order derivatives passed to the
        constructor will be ignored.

        second_derivative is the rate of the rate of change in the
        radius with respect to the parameter.  If the curve is
        understood to be a particle flying through space in time, the
        second derivative would be its acceleration.  What's passed in
        is the same as first_derivative.  Should you opt for passing in
        None, you will not be able to query the normal or binormal
        vectors, nor the curvature or torsion.

        third_derivative is the thrice iterated derivative with respect
        to parameter of the radius.  If the curve is to be understood as
        economic price level (somehow in 3D), then Richard Nixon would
        advertise that the third derivative is negative in saying "the
        rate of increase of inflation was decreasing" in hopes of
        deceiving you that the economy is good.  The shape of this
        argument is just the same as for first_derivative and
        second_derivative, but with None passed in, only torsion is off
        limits.
        """
        self._radius = radius
        self._first_derivative = first_derivative
        self._second_derivative = second_derivative
        self._third_derivative = third_derivative

    def radius_at(self, parameter: Number) -> npt.ArrayLike:
        """Computes the 3-vector coordinate at parameter and returns it."""
        return self._radius(parameter)

    def first_derivative_at(self, parameter: Number) -> npt.ArrayLike:
        """Computes the 3-vector first derivative at parameter and
        returns it.
        """
        return self._first_derivative(parameter)

    def second_derivative_at(self, parameter: Number) -> npt.ArrayLike:
        """Computes the 3-vector second derivative at parameter and
        returns it.
        """
        return self._second_derivative(parameter)

    def third_derivative_at(self, parameter: Number) -> npt.ArrayLike:
        """Computes the 3-vector third derivative at parameter and
        returns it.
        """
        return self._third_derivative(parameter)

    def curvature_at(self, parameter: Number) -> float:
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
        return np.inf
        # return (
        #     np.linalg.norm(
        #         np.cross(
        #             self._first_derivative(parameter),
        #             self._second_derivative(parameter),
        #         )
        #     )
        #     / np.linalg.norm(self._first_derivative(parameter)) ** 3
        # )

    def torsion_at(self, parameter: Number) -> float:
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
        return np.inf

    def tangent_vector_at(self, parameter: Number) -> float:
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
        return np.array([np.inf, np.inf, np.inf])

    def normal_vector_at(self, parameter: Number) -> float:
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
        """
        return np.array([np.inf, np.inf, np.inf])

    def binormal_vector_at(self, parameter: Number) -> float:
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
        return np.array([np.inf, np.inf, np.inf])
