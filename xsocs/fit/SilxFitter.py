#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
__date__ = "01/01/2017"
__license__ = "MIT"


import numpy as np

from silx.math.fit import fittheories, sum_agauss
from silx.math.fit.fitmanager import FitManager

from .Plotter import Plotter

from ..process.Fitter import Fitter
from ..process.fitresults import FitStatus


class SilxFitter(Fitter):
    p_types = ['A', 'P', 'F']

    def __init__(self, *args, **kwargs):

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
            self._shared_results.set_qz_results(i_fit, results, FitStatus.OK)
        else:
            self._shared_results.set_qz_results(i_fit, results,
                                                FitStatus.FAILED)


class SilxPlotter(Plotter):
    def plotFit(self, plot, x, peakParams):
        for peakName, peak in peakParams.items():
            area = peak.get('area')
            position = peak.get('position')
            width = peak.get('fwhm')

            params = [area, position, width]

            fitSum = None

            if np.all(np.isfinite(params)):
                fitted = sum_agauss(x, *params)
                plot.addCurve(x,
                              fitted,
                              legend='{0}'.format(peakName))
                if fitSum is None:
                    fitSum = fitted
                else:
                    fitSum += fitted

            if fitSum is not None:
                plot.addCurve(x, fitSum, legend='Sum')

    def getPlotTitle(self):
        return 'Silx Gaussian Fit'


if __name__ == '__main__':
    pass
