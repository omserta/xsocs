# coding: utf-8
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
__license__ = "MIT"
__date__ = "15/09/2016"


import numpy as np


from silx.math.fit import sum_agauss

from ....process.fit_funcs import gaussian


# TODO : allow users to register plot functions associated with the kind
# of process results that are being displayed
def plotGaussian(plots, index, fitH5,
                 entry,
                 xAcqQX, xAcqQY, xAcqQZ,
                 yAcqQX, yAcqQY, yAcqQZ):
    """
    Plots the "Gaussian" fit results
    :param plots: plot widgets
    :param index: index of the selected point (in the results array)
    :param fitH5: instance of FitH5. This instance may be already opened by
        the caller.
    :param entry: name of the entry to plot
    :param xData: x axis values of the fitted data
    :param acqX: x axis values of the acquired data
    :param acqY: y axis values of the acquired data
    :return:
    """

    plots[0].addCurve(xAcqQX, yAcqQX, legend='measured')
    plots[0].setGraphTitle('QX / Gaussians')

    plots[1].addCurve(xAcqQY, yAcqQY, legend='measured')
    plots[1].setGraphTitle('QY / Gaussians')

    plots[2].addCurve(xAcqQZ, yAcqQZ, legend='measured')
    plots[2].setGraphTitle('QZ / Gaussians')

    with fitH5:
        xFitQX = fitH5.get_qx(entry)
        xFitQY = fitH5.get_qy(entry)
        xFitQZ = fitH5.get_qz(entry)

        sumX = None
        sumY = None
        sumZ = None

        processes = fitH5.processes(entry)

        for process in processes:
            heights = fitH5.get_result(entry, process, 'intensity')
            positions = fitH5.get_result(entry, process, 'position')
            widths = fitH5.get_result(entry, process, 'width')

            h_x = heights.qx[index]
            p_x = positions.qx[index]
            w_x = widths.qx[index]

            h_y = heights.qy[index]
            p_y = positions.qy[index]
            w_y = widths.qy[index]

            h_z = heights.qz[index]
            p_z = positions.qz[index]
            w_z = widths.qz[index]

            params = [h_x, p_x, w_x]
            if np.all(np.isfinite(params)):
                fitted = gaussian(xFitQX, *params)
                plots[0].addCurve(xFitQX, fitted,
                                  legend='QX process={0}'.format(process))
                if sumX is None:
                    sumX = fitted
                else:
                    sumX += fitted

            params = [h_y, p_y, w_y]
            if np.all(np.isfinite(params)):
                fitted = gaussian(xFitQY, *params)
                plots[1].addCurve(xFitQY, fitted,
                                  legend='QY process={0}'.format(process))
                if sumY is None:
                    sumY = fitted
                else:
                    sumY += fitted

            params = [h_z, p_z, w_z]
            if np.all(np.isfinite(params)):
                fitted = gaussian(xFitQZ, *params)
                plots[2].addCurve(xFitQZ, fitted,
                                  legend='QZ process={0}'.format(process))
                if sumZ is None:
                    sumZ = fitted
                else:
                    sumZ += fitted

        if sumX is not None:
            plots[0].addCurve(xFitQX, sumX, legend='Sum')

        if sumY is not None:
            plots[1].addCurve(xFitQY, sumY, legend='Sum')

        if sumZ is not None:
            plots[2].addCurve(xFitQZ, sumZ, legend='Sum')


# TODO : allow users to register plot functions associated with the kind
# of process results that are being displayed
def plotSilx(plots, index, fitH5,
             entry,
             xAcqQX, xAcqQY, xAcqQZ,
             yAcqQX, yAcqQY, yAcqQZ):
    """
    Plots the "Gaussian" fit results
    :param plots: plot widgets
    :param index: index of the selected point (in the results array)
    :param fitH5: instance of FitH5. This instance may be already opened by
        the caller.
    :param entry: name of the entry to plot
    :param xData: x axis values of the fitted data
    :param acqX: x axis values of the acquired data
    :param acqY: y axis values of the acquired data
    :return:
    """

    plots[0].addCurve(xAcqQX, yAcqQX, legend='measured')
    plots[0].setGraphTitle('QX / Gaussians')

    plots[1].addCurve(xAcqQY, yAcqQY, legend='measured')
    plots[1].setGraphTitle('QY / Gaussians')

    plots[2].addCurve(xAcqQZ, yAcqQZ, legend='measured')
    plots[2].setGraphTitle('QZ / Gaussians')

    with fitH5:
        xFitQX = fitH5.get_qx(entry)
        xFitQY = fitH5.get_qy(entry)
        xFitQZ = fitH5.get_qz(entry)

        paramsX = []
        paramsY = []
        paramsZ = []

        processes = fitH5.processes(entry)

        for process in processes:
            heights = fitH5.get_result(entry, process, 'intensity')
            positions = fitH5.get_result(entry, process, 'position')
            widths = fitH5.get_result(entry, process, 'fwhm')

            h_x = heights.qx[index]
            p_x = positions.qx[index]
            w_x = widths.qx[index]

            h_y = heights.qy[index]
            p_y = positions.qy[index]
            w_y = widths.qy[index]

            h_z = heights.qz[index]
            p_z = positions.qz[index]
            w_z = widths.qz[index]

            params = [h_x, p_x, w_x]
            if np.all(np.isfinite(params)):
                fitted = sum_agauss(xFitQX, *params)
                plots[0].addCurve(xFitQX, fitted,
                                  legend='QX process={0}'.format(process))
                paramsX.extend(params)

            params = [h_y, p_y, w_y]
            if np.all(np.isfinite(params)):
                fitted = sum_agauss(xFitQY, *params)
                plots[1].addCurve(xFitQY, fitted,
                                  legend='QY process={0}'.format(process))
                paramsY.extend(params)

            params = [h_z, p_z, w_z]
            if np.all(np.isfinite(params)):
                fitted = sum_agauss(xFitQZ, *params)
                plots[2].addCurve(xFitQZ, fitted,
                                  legend='QZ process={0}'.format(process))
                paramsZ.extend(params)

        if paramsX:
            plots[0].addCurve(xAcqQX,
                              sum_agauss(xAcqQX, *paramsX),
                              legend='Sum')

        if paramsY:
            plots[1].addCurve(xAcqQY,
                              sum_agauss(xAcqQY, *paramsY),
                              legend='Sum')

        if paramsZ:
            plots[2].addCurve(xAcqQZ,
                              sum_agauss(xAcqQZ, *paramsZ),
                              legend='Sum')


def plotCentroid(plots, index, fitH5,
                 entry,
                 xAcqQX, xAcqQY, xAcqQZ,
                 yAcqQX, yAcqQY, yAcqQZ):
    """
    Plot the results from a "centroid" fit.
    :param plots: the plot widgets
    :param index: index of the sample point
    :param fitH5: fitH5 file
    :param entry: name of the entry in the fitH5
    :param process: name of the process in the fitH5
    :param xAcqQX: measured Qx data, x axis
    :param xAcqQY: measured Qy data, x axis
    :param xAcqQZ: measured Qz data, x axis
    :param yAcqQX: measured Qx data, y axis
    :param yAcqQY:measured Qy data, y axis
    :param yAcqQZ:measured Qz data, y axis
    :return:
    """

    # TODO : put all this in a toolbox, so it can be shared between
    # the plot and the fit functions

    process = fitH5.processes(entry)[0]

    positions = fitH5.get_result(entry, process, 'COM')

    plots[0].addCurve(xAcqQX, yAcqQX, legend='measured')
    plots[0].addXMarker(positions.qx[index], legend='center of mass')
    plots[0].setGraphTitle('QX center of mass')

    plots[1].addCurve(xAcqQY, yAcqQY, legend='measured')
    plots[1].addXMarker(positions.qy[index], legend='center of mass')
    plots[1].setGraphTitle('QY center of mass')

    plots[2].addCurve(xAcqQZ, yAcqQZ, legend='measured')
    plots[2].addXMarker(positions.qz[index], legend='center of mass')
    plots[2].setGraphTitle('QZ center of mass')


if __name__ == '__main__':
    pass
