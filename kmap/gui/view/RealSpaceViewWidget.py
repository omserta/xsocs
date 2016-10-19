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

try:
    from silx.gui.plot.ImageRois import ImageRoiManager
except ImportError:
    # TODO remove this import once the ROIs are added to the silx release.
    from ..silx_imports.ImageRois import ImageRoiManager


class RectRoiWidget(Qt.QWidget):
    sigRoiApplied = Qt.Signal(object)

    def __init__(self, roiManager, parent=None):
        # TODO :
        # support multiple ROIs then batch them
        super(RectRoiWidget, self).__init__(parent)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_DialogCloseButton)
        self.__discardBn = discardBn = Qt.QToolButton()
        discardBn.setToolTip('Discard ROI')
        discardBn.setStatusTip('Discard ROI')
        discardBn.setIcon(icon)
        discardBn.setEnabled(False)
        layout.addWidget(discardBn)
        discardBn.clicked.connect(self.__discardRoi)

        icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        self.__applyBn = applyBn = Qt.QToolButton()
        applyBn.setToolTip('Apply ROI')
        applyBn.setStatusTip('Apply ROI')
        applyBn.setIcon(icon)
        applyBn.setEnabled(False)
        layout.addWidget(applyBn)
        applyBn.clicked.connect(self.__applyRoi)

        layout.addWidget(Qt.QLabel('x='))
        self._xEdit = edit = Qt.QLineEdit()
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        layout.addWidget(Qt.QLabel('y='))
        self._yEdit = edit = Qt.QLineEdit()
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        self._wEdit = edit = Qt.QLineEdit()
        layout.addWidget(Qt.QLabel('w='))
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        self._hEdit = edit = Qt.QLineEdit()
        layout.addWidget(Qt.QLabel('h='))
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)

        fm = edit.fontMetrics()
        padding = 2 * fm.width('0')
        text = '0' * 8
        width = fm.width(text) + padding
        self._xEdit.setFixedWidth(width)
        self._yEdit.setFixedWidth(width)
        self._wEdit.setFixedWidth(width)
        self._hEdit.setFixedWidth(width)
        layout.setSizeConstraint(Qt.QLayout.SetMinimumSize)

        # TODO : weakref
        self.__roiManager = roiManager
        roiManager.sigRoiDrawingFinished.connect(self.__roiDrawingFinished,
                                                 Qt.Qt.QueuedConnection)
        roiManager.sigRoiRemoved.connect(self.__roiRemoved,
                                         Qt.Qt.QueuedConnection)
        roiManager.sigRoiMoved.connect(self.__roiMoved,
                                       Qt.Qt.QueuedConnection)

    def __discardRoi(self, checked):
        self.__roiManager.clear()

    def __applyRoi(self, checked):
        # At the moment we only support one roi at a time.
        roi = self.__roiManager.rois
        roiItem = self.__roiManager.roiItem(roi[0])
        xMin = roiItem.pos[0]
        xMax = xMin + roiItem.width
        yMin = roiItem.pos[1]
        yMax = yMin + roiItem.height
        self.sigRoiApplied.emit([xMin, xMax, yMin, yMax])

    def __roiDrawingFinished(self, event):
        self.__display(event['xdata'], event['ydata'])
        self.__discardBn.setEnabled(True)
        self.__applyBn.setEnabled(True)

    def __clear(self):
        self._xEdit.clear()
        self._yEdit.clear()
        self._wEdit.clear()
        self._hEdit.clear()

    def __display(self, xData, yData):
        xMin, xMax = xData[0], xData[2]
        if xMax < xMin:
            xMin, xMax = xMax, xMin
        yMin, yMax = yData[0], yData[1]
        if yMax < yMin:
            yMin, yMax = yMax, yMin
        self._xEdit.setText(str(xMin))
        self._yEdit.setText(str(yMin))
        self._wEdit.setText(str(xMax - xMin))
        self._hEdit.setText(str(yMax - yMin))

    def __roiRemoved(self, name):
        self.__clear()
        self.__discardBn.setEnabled(False)
        self.__applyBn.setEnabled(False)

    def __roiMoved(self, event):
        self.__display(event['xdata'], event['ydata'])


class RealSpaceViewWidgetEvent(DataViewEvent):
    pass


class RealSpaceViewWidget(DataViewWidget):

    plot = property(lambda self: self.__plotWindow)

    def __init__(self, index, parent=None, **kwargs):
        super(RealSpaceViewWidget, self).__init__(index, parent=parent)

        self.__plotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

        self.__roiManager = roiManager = ImageRoiManager(plotWindow)
        roiToolBar = roiManager.toolBar(rois=['rectangle'],
                                        options=['show'])
        roiToolBar.addSeparator()

        rectRoiWidget = RectRoiWidget(roiManager)
        roiToolBar.addWidget(rectRoiWidget)
        rectRoiWidget.sigRoiApplied.connect(self.__roiApplied)

        plotWindow.addToolBarBreak()
        plotWindow.addToolBar(roiToolBar)

        self.setCentralWidget(plotWindow)

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

    def __roiApplied(self, roi):
        self._emitEvent(RealSpaceViewWidgetEvent(self, roi))


if __name__ == '__main__':
    pass
