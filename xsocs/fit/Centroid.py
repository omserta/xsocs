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

from .Plotter import Plotter

from ..process.Fitter import Fitter
from ..process.fitresults import FitStatus


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


class CentroidPlotter(Plotter):



    def plotFit(self, plot, x, peakParams):
        plot.setGraphTitle('QX center of mass')
        for peakName, peak in peakParams.items():
            center = peak.get('COM')

            if np.isfinite(center):
                plot.addXMarker(center, legend='center of mass')

    def getPlotTitle(self):
        return 'Center Of Mass'

# process = fitH5.processes(entry)[0]
#
#     positions = fitH5.get_result(entry, process, 'COM')
#
#     plots[0].addCurve(xAcqQX, yAcqQX, legend='measured')
#     plots[0].addXMarker(positions.qx[index], legend='center of mass')
#     plots[0].setGraphTitle('QX center of mass')
#
#     plots[1].addCurve(xAcqQY, yAcqQY, legend='measured')
#     plots[1].addXMarker(positions.qy[index], legend='center of mass')
#     plots[1].setGraphTitle('QY center of mass')
#
#     plots[2].addCurve(xAcqQZ, yAcqQZ, legend='measured')
#     plots[2].addXMarker(positions.qz[index], legend='center of mass')
#     plots[2].setGraphTitle('QZ center of mass')
#


if __name__ == '__main__':
    pass
