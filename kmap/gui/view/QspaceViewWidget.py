# coding: utf-8
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
__license__ = "MIT"
__date__ = "15/09/2016"

import numpy as np
from matplotlib import cm

from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow

from .DataViewWidget import DataViewWidget, DataViewEvent


class QSpaceViewWidgetEvent(DataViewEvent):
    pass


class QSpaceViewWidget(DataViewWidget):

    plot = property(lambda self: self.__plotWindow)

    def __init__(self, index, parent=None, **kwargs):
        super(QSpaceViewWidget, self).__init__(index, parent=parent)

        self.__plotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)
        self.setCentralWidget(plotWindow)

    # TODO : refactor this in a common base with RealSpaceViewWidget
    def setPlotData(self, x, y, data):
        plot = self.__plotWindow
        if data.ndim == 1:
            # scatter
            min_, max_ = data.min(), data.max()
            colormap = cm.jet
            colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
            plot.addCurve(x, y,
                          color=colors,
                          symbol='s',
                          linestyle='')
        elif data.ndim == 2:
            # image
            min_, max_ = data.min(), data.max()
            colormap = {'name': 'temperature',
                        'normalization': 'linear',
                        'autoscale': True,
                        'vmin': min_,
                        'vmax': max_}
            origin = x[0], y[0]
            scale = (x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y)
            plot.addImage(data,
                          origin=origin,
                          scale=scale,
                          colormap=colormap)
        else:
            raise ValueError('data has {0} dimensions, expected 1 or 2.'
                             ''.format(data.ndim))