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

from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow

from matplotlib import cm

from ..model.TreeView import TreeView
from ..model.ModelDef import ModelColumns, ModelRoles
from ..project.XsocsH5Factory import h5NodeToProjectItem
from ..widgets.XsocsPlot2D import XsocsPlot2D
from ..widgets.Input import StyledLineEdit
from ..widgets.Containers import GroupBox

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

        self.__roiToolBar = roiToolBar = roiManager.toolBar(rois=['rectangle'],
                                        options=['show'])
        roiToolBar.setMovable(False)

        topLayout = Qt.QVBoxLayout(self)

        grpBox = GroupBox('ROI')
        layout = Qt.QGridLayout(grpBox)

        row = 0
        layout.addWidget(roiToolBar, row, 0, 1, 2, Qt.Qt.AlignTop)

        row += 1
        self._xEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('x='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._yEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('y='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._wEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('w='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._hEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('h='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1

        hLayout = Qt.QHBoxLayout()

        style = Qt.QApplication.style()

        icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        self.__applyBn = applyBn = Qt.QToolButton()
        applyBn.setToolTip('Apply ROI')
        applyBn.setStatusTip('Apply ROI')
        applyBn.setIcon(icon)
        applyBn.setToolButtonStyle(Qt.Qt.ToolButtonTextBesideIcon)
        applyBn.setText('To Q Space')
        applyBn.setEnabled(False)
        hLayout.addWidget(applyBn)
        applyBn.clicked.connect(self.__applyRoi)

        icon = style.standardIcon(Qt.QStyle.SP_DialogCloseButton)
        self.__discardBn = discardBn = Qt.QToolButton()
        discardBn.setToolTip('Discard ROI')
        discardBn.setStatusTip('Discard ROI')
        discardBn.setIcon(icon)
        discardBn.setEnabled(False)
        hLayout.addWidget(discardBn, Qt.Qt.AlignRight)
        discardBn.clicked.connect(self.__discardRoi)

        layout.addLayout(hLayout, row, 0, 1, 2, Qt.Qt.AlignCenter)

        # topLayout.setSizeConstraint(Qt.QLayout.SetMinimumSize)

        topLayout.addWidget(grpBox)
        topLayout.addStretch(100)

        # TODO : weakref
        self.__roiManager = roiManager
        roiManager.sigRoiDrawingFinished.connect(self.__roiDrawingFinished,
                                                 Qt.Qt.QueuedConnection)
        roiManager.sigRoiRemoved.connect(self.__roiRemoved,
                                         Qt.Qt.QueuedConnection)
        roiManager.sigRoiMoved.connect(self.__roiMoved,
                                       Qt.Qt.QueuedConnection)

    def sizeHint(self):
        return Qt.QSize(self.__roiToolBar.sizeHint().width() + 10, 0)

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


class IntensityTree(TreeView):
    sigCurrentChanged = Qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super(IntensityTree, self).__init__(*args, **kwargs)
        self.disableDelegateForColumn(1, True)
        for col in range(ModelColumns.ColumnMax):
            if col != ModelColumns.NameColumn:
                self.setColumnHidden(col, True)

    def currentChanged(self, current, previous):
        super(IntensityTree, self).currentChanged(current, previous)
        node = current.data(ModelRoles.InternalDataRole)
        if not node:
            return
        projectItem = h5NodeToProjectItem(node)
        self.sigCurrentChanged.emit(projectItem)


class IntensityView(Qt.QMainWindow):
    sigProcessApplied = Qt.Signal(object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(IntensityView, self).__init__(parent=parent)

        self.setWindowTitle('[XSOCS] {0}'.format(node.h5Path))

        self.__plotWindow = plotWindow = XsocsPlot2D()
        plotWindow.setShowMousePosition(True)
        plotWindow.setShowSelectedCoordinates(False)

        dock = Qt.QDockWidget(self)
        tree = IntensityTree(self, model=model)
        index = node.index()
        tree.setRootIndex(index)
        tree.sigCurrentChanged.connect(self.__itemSelected)
        dock.setWidget(tree)
        features = dock.features() ^ Qt.QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, dock)

        self.__roiManager = roiManager = ImageRoiManager(plotWindow)
        # roiToolBar = roiManager.toolBar(rois=['rectangle'],
        #                                 options=['show'])
        # roiToolBar.addSeparator()
        # plotWindow.addToolBarBreak()
        # plotWindow.addToolBar(roiToolBar)

        rectRoiWidget = RectRoiWidget(roiManager)
        rectRoiWidget.sigRoiApplied.connect(self.__roiApplied)


        dock = Qt.QDockWidget(self)
        dock.setWidget(rectRoiWidget)
        features = dock.features() ^ Qt.QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, dock)

        self.setCentralWidget(plotWindow)

    def __itemSelected(self, item):
        intensity, positions = item.getScatterData()
        self.setPlotData(positions.pos_0, positions.pos_1, intensity)

    def setPlotData(self, x, y, data):
        plot = self.__plotWindow
        plot.setPlotData(x, y, data)
        # if data.ndim == 1:
        #     # scatter
        #     min_, max_ = data.min(), data.max()
        #     colormap = cm.jet
        #     colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
        #     plot.addCurve(x, y,
        #                   color=colors,
        #                   symbol='s',
        #                   linestyle='')
        # elif data.ndim == 2:
        #     # image
        #     min_, max_ = data.min(), data.max()
        #     colormap = {'name': 'temperature',
        #                 'normalization': 'linear',
        #                 'autoscale': True,
        #                 'vmin': min_,
        #                 'vmax': max_}
        #     origin = x[0], y[0]
        #     scale = (x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y)
        #     plot.addImage(data,
        #                   origin=origin,
        #                   scale=scale,
        #                   colormap=colormap)
        # else:
        #     raise ValueError('data has {0} dimensions, expected 1 or 2.'
        #                      ''.format(data.ndim))

    def __roiApplied(self, roi):
        self.sigProcessApplied.emit(roi)


if __name__ == '__main__':
    pass
