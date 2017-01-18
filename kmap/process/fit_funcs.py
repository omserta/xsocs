#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["D. Naudet"]
__date__ = "01/06/2016"
__license__ = "MIT"

import numpy as np

from scipy.optimize import leastsq

# Some constants
_const_inv_2_pi_ = np.sqrt(2 * np.pi)


# 1d Gaussian func
# TODO : optimize
def gaussian(x, a, c, s):
    """
    Returns (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    :param x: values for which the gaussian must be computed
    :param a: area under curve ( amplitude * s * sqrt(2 * pi) )
    :param c: center
    :param stdev: standard deviation
    :return: (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    """
    return (a * (1. / (_const_inv_2_pi_ * s)) *
            np.exp(-0.5 * ((x - c) / s) ** 2))


# 1d Gaussian fit
# TODO : optimize
def gaussian_fit_err(p, x, y):
    """

    :param p:
    :param x:
    :param y:
    :return:
    """
    return gaussian(x, *p) - y


def gaussian_fit(x, y, p):
    """
    Fits (leastsq) a gaussian on the provided data f(x) = y.
    p = (a, c, s)
    and f(x) = (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    :param x:
    :param y:
    :param p:
    :return:
    """
    result = leastsq(gaussian_fit_err,
                     p,
                     args=(x, y,),
                     maxfev=100000,
                     full_output=True)

    if result[4] not in [1, 2, 3, 4]:
        raise ValueError('Failed to fit : {0}.'.format(result[3]))

    return result[0]


def centroid(x, y, p):
    """
    Computes the center of mass of the provided data.
    Returns the value closest to the center of mass, and the
    the center of mass
    :param x:
    :param y:
    :param p:
    :return: list
    """
    # TODO : throw exception if fit failed
    com = x.dot(y) / y.sum()
    idx = np.abs(x - com).argmin()
    return [y[idx], com, np.nan]
