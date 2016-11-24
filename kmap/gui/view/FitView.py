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
from ..model.TreeView import TreeView
from ..model.Model import Model
from ..model.ModelDef import ModelColumns, ModelRoles
from ..project.XsocsH5Factory import h5NodeToProjectItem
from ..project.FitGroup import FitItem
from ...io.FitH5 import FitH5



class FitTree(TreeView):
    sigCurrentChanged = Qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super(FitTree, self).__init__(*args, **kwargs)
        self.disableDelegateForColumn(1, True)
        for col in range(ModelColumns.ColumnMax):
            if col != ModelColumns.NameColumn:
                self.setColumnHidden(col, True)

    def currentChanged(self, current, previous):
        node = current.data(ModelRoles.InternalDataRole)
        dataType = node.nodeName
        fitItem = h5NodeToProjectItem(node.parent())
        if not isinstance(fitItem, FitItem):
            return
        self.sigCurrentChanged.emit({'event': dataType, 'item': fitItem})


class FitView(Qt.QMainWindow):
    sigProcessApplied = Qt.Signal(object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(FitView, self).__init__(parent=parent)

        treeDock = Qt.QDockWidget(self)
        features = treeDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        treeDock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, treeDock)

        self.__xPlotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

        xDock = Qt.QDockWidget()
        xDock.setWidget(plotWindow)
        features = xDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        xDock.setFeatures(features)
        self.splitDockWidget(treeDock, xDock, Qt.Qt.Vertical)

        self.__yPlotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                     curveStyle=False,
                                                     mask=False,
                                                     roi=False,
                                                     **kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

        yDock = Qt.QDockWidget()
        yDock.setWidget(plotWindow)
        features = yDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        yDock.setFeatures(features)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, yDock)

        self.__zPlotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                     curveStyle=False,
                                                     mask=False,
                                                     roi=False,
                                                     **kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

        zDock = Qt.QDockWidget()
        zDock.setWidget(plotWindow)
        features = zDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        zDock.setFeatures(features)
        self.splitDockWidget(yDock, zDock, Qt.Qt.Vertical)

        tree = FitTree(self, model=model)
        index = node.index()
        tree.setRootIndex(index)
        tree.sigCurrentChanged.connect(self.__itemSelected)
        treeDock.setWidget(tree)

    def __itemSelected(self, event):
        dtype = event['event']
        item = event['item']

        with item.fitH5 as fitH5:
            sampleX, sampleY = fitH5.scan_positions()

            if dtype == 'height':
                xData = fitH5.x_height()
                yData = fitH5.y_height()
                zData = fitH5.z_height()
            elif dtype == 'center':
                xData = fitH5.x_center()
                yData = fitH5.y_center()
                zData = fitH5.z_center()
            elif dtype == 'width':
                xData = fitH5.x_width()
                yData = fitH5.y_width()
                zData = fitH5.z_width()
            else:
                raise ValueError('Unknown event {0}.'.format(dtype))

        self.__setPlotData(self.__xPlotWindow, sampleX, sampleY, xData)
        self.__setPlotData(self.__yPlotWindow, sampleX, sampleY, yData)
        self.__setPlotData(self.__zPlotWindow, sampleX, sampleY, zData)

    def __setPlotData(self, plot, x, y, data):
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

    def __roiApplied(self, roi):
        self.sigProcessApplied.emit(roi)


if __name__ == '__main__':
    pass
