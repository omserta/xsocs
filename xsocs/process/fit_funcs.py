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
import time

from scipy.optimize import leastsq

from .fitresults import FitStatus

# Some constants
_const_inv_2_pi_ = np.sqrt(2 * np.pi)


class Fitter(object):
    def __init__(self, qx, qy, qz,
                 shared_results):
        super(Fitter, self).__init__()

        self._shared_results = shared_results
        self._qx = qx
        self._qy = qy
        self._qz = qz

    def fit(self, i_fit, i_cube, qx_profile, qy_profile, qz_profile):
        raise NotImplementedError('Not implemented.')


class GaussianFitter(Fitter):

    def ptime(self):
        print(self.t_set, self.t_estim, self.t_fit, self.t_parse, self.t_write)

    def __init__(self, *args, **kwargs):
        super(GaussianFitter, self).__init__(*args, **kwargs)

        self._n_peaks = self._shared_results._n_peaks
        self._z_0 = np.tile([1.0, self._qz.mean(), 1.0],
                            self._shared_results._n_peaks)
        self._y_0 = np.tile([1.0, self._qy.mean(), 1.0],
                            self._shared_results._n_peaks)
        self._x_0 = np.tile([1.0, self._qx.mean(), .1],
                            self._shared_results._n_peaks)

        self.t_set = 0.
        self.t_estim = 0.
        self.t_fit = 0.
        self.t_write = 0.
        self.t_parse = 0.

    def fit(self, i_fit, i_cube, qx_profile, qy_profile, qz_profile):

        z_fit, success_z = gaussian_fit(self._qz, qz_profile, self._z_0)
        y_fit, success_y = gaussian_fit(self._qy, qy_profile, self._y_0)
        t0 = time.time()
        x_fit, success_x = gaussian_fit(self._qx, qx_profile, self._x_0)
        self.t_fit += time.time() - t0

        self._shared_results.set_qz_results(i_fit, z_fit, success_z)
        self._shared_results.set_qy_results(i_fit, y_fit, success_y)

        t0 = time.time()
        self._shared_results.set_qx_results(i_fit, x_fit, success_x)
        self.t_write += time.time() - t0


class SilxFitter(Fitter):
    p_types = ['A', 'P', 'F']

    def __init__(self, *args, **kwargs):
        from silx.math.fit import fittheories
        from silx.math.fit.fitmanager import FitManager

        super(SilxFitter, self).__init__(*args, **kwargs)

        self._n_peaks = self._shared_results._n_peaks

        self._fit = FitManager()
        self._fit.loadtheories(fittheories)
        self._fit.settheory('Area Gaussians')
        self._results = np.zeros(3 * self._n_peaks)

    def fit(self, i_fit, i_cube, qx_profile, qy_profile, qz_profile):

        fit = self._fit
        results = self._results
        failed = False

        fit.setdata(x=self._qx, y=qx_profile)

        try:
            fit.estimate()
            fit.runfit()
        except Exception as ex:
            failed = True

        results[:] = np.nan

        if not failed:

            for param in fit.fit_results:
                p_name = param['name']
                p_type = p_name[0]
                peak_idx = int(p_name[-1]) - 1

                if peak_idx >= self._n_peaks:
                    continue

                # TODO : error management
                param_idx = self.p_types.index(p_type)
                results[peak_idx * 3 + param_idx] = param['fitresult']

            self._shared_results.set_qx_results(i_fit, results, FitStatus.OK)
        else:
            self._shared_results.set_qx_results(i_fit, results,
                                                FitStatus.FAILED)

        failed = False
        fit.setdata(x=self._qy, y=qy_profile)

        try:
            fit.estimate()
            fit.runfit()
        except Exception as ex:
            failed = True

        results[:] = np.nan

        if not failed:
            for param in fit.fit_results:
                p_name = param['name']
                p_type = p_name[0]
                peak_idx = int(p_name[-1]) - 1

                if peak_idx >= self._n_peaks:
                    continue

                # TODO : error management
                param_idx = self.p_types.index(p_type)
                results[peak_idx * 3 + param_idx] = param['fitresult']
            self._shared_results.set_qy_results(i_fit, results, FitStatus.OK)
        else:
            self._shared_results.set_qy_results(i_fit, results,
                                                FitStatus.FAILED)

        failed = False
        fit.setdata(x=self._qz, y=qz_profile)

        try:
            fit.estimate()
            fit.runfit()
        except:
            failed = True

        results[:] = np.nan

        if failed:
            for param in fit.fit_results:
                p_name = param['name']
                p_type = p_name[0]
                peak_idx = int(p_name[-1]) - 1

                if peak_idx >= self._n_peaks:
                    continue

                # TODO : error management
                param_idx = self.p_types.index(p_type)
                results[peak_idx * 3 + param_idx] = param['fitresult']
            self._shared_results.set_qz_results(i_fit, results, FitStatus.OK)
        else:
            self._shared_results.set_qz_results(i_fit, results,
                                                FitStatus.FAILED)


class CentroidFitter(Fitter):
    def fit(self, i_fit, i_cube, qx_profile, qy_profile, qz_profile):

        zSum = qz_profile.sum()
        if sum != 0:
            com = self._qz.dot(qz_profile) / zSum
            idx = np.abs(self._qz - com).argmin()
            i_max = qz_profile.max()
            self._shared_results.set_qz_results(i_fit,
                                                [qz_profile[idx], com, i_max],
                                                FitStatus.OK)
        else:
            self._shared_results.set_qz_results(i_fit,
                                                [np.nan, np.nan, np.nan],
                                                FitStatus.FAILED)

        ySum = qy_profile.sum()
        if ySum != 0:
            com = self._qy.dot(qy_profile) / ySum
            idx = np.abs(self._qy - com).argmin()
            i_max = qy_profile.max()
            self._shared_results.set_qy_results(i_fit,
                                                [qy_profile[idx], com, i_max],
                                                FitStatus.OK)
        else:
            self._shared_results.set_qy_results(i_fit,
                                                [np.nan, np.nan, np.nan],
                                                FitStatus.FAILED)

        xSum = qx_profile.sum()
        if xSum != 0:
            com = self._qx.dot(qx_profile) / xSum
            idx = np.abs(self._qx - com).argmin()
            i_max = qx_profile.max()
            self._shared_results.set_qx_results(i_fit,
                                                [qx_profile[idx], com, i_max],
                                                FitStatus.OK)
        else:
            self._shared_results.set_qx_results(i_fit,
                                                [np.nan, np.nan, np.nan],
                                                FitStatus.FAILED)
        

# 1d Gaussian func
# TODO : optimize
def gaussian(x, a, c, s):
    """
    Returns (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    :param x: values for which the gaussian must be computed
    :param a: area under curve ( amplitude * s * sqrt(2 * pi) )
    :param c: center
    :param s: sigma
    :return: (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    """
    # s /= _two_sqrt_2_log_2
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
    n_p = len(p) // 3
    gSum = 0.
    for i_p in range(n_p):
        gSum += gaussian(x, *p[i_p*3:i_p*3 + 3])
    return gSum - y
    # return gaussian(x, *p) - y


_two_sqrt_2_log_2 = 2 * np.sqrt(2 * np.log(2))


def gaussian_fit(x, y, p):
    """
    Fits (leastsq) a gaussian on the provided data f(x) = y.
    p = (a, c, s)
    and f(x) = (a / (sqrt(2 * pi) * s)) * exp(- 0.5 * ((x - c) / s)^2)
    :param x:
    :param y:
    :param p:
    :return: area, position, fwhm
    """
    result = leastsq(gaussian_fit_err,
                     p,
                     args=(x, y,),
                     maxfev=100000,
                     full_output=True)

    if result[4] not in [1, 2, 3, 4]:
        return [np.nan] * len(p), FitStatus.FAILED

    # results = result[0]
    # results[2] *= _two_sqrt_2_log_2

    return result[0], FitStatus.OK


def _gauss_first_guess(x, y):
    i_max = y.argmax()
    y_max = y[i_max]
    p1 = x[i_max]
    i_fwhm = np.where(y >= y_max / 2.)[0]
    fwhm = (x[1] - x[0]) * len(i_fwhm)
    p2 = fwhm / np.sqrt(2 * np.log(2))  # 2.35482
    p0 = y_max * np.sqrt(2 * np.pi) * p2
    return [p0, p1, p2]


if __name__ == '__main__':
    pass
