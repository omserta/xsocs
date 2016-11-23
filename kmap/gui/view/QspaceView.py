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
from plot3d.ScalarFieldView import ScalarFieldView
from plot3d.SFViewParamTree import TreeView as SFTreeView

from ...io.QSpaceH5 import QSpaceH5
from ..model.ModelDef import ModelRoles
# from ..project.HybridItem import HybridItem
from ..project.QSpaceGroup import QSpaceItem
from ..model.TreeView import TreeView
from ..project.XsocsH5Factory import h5NodeToProjectItem


class QSpaceTree(TreeView):
    pass
    # sigCurrentChanged = Qt.Signal(object)

    # def __init__(self, *args, **kwargs):
    #     super(IntensityTree, self).__init__(*args, **kwargs)
    #     self.disableDelegateForColumn(1, True)
    #     for col in range(ModelColumns.ColumnMax):
    #         if col != ModelColumns.NameColumn:
    #             self.setColumnHidden(col, True)
    #
    # def currentChanged(self, current, previous):
    #     node = current.data(ModelRoles.InternalDataRole)
    #     if not node:
    #         return
    #     projectItem = h5NodeToProjectItem(node)
    #     self.sigCurrentChanged.emit(projectItem)


class QSpaceView(Qt.QMainWindow):
    sigProcessApplied = Qt.Signal(object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(QSpaceView, self).__init__(parent)

        self.__plotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)

        plotWindow.sigPlotSignal.connect(self.__plotSignal)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)
        item = h5NodeToProjectItem(node)
        with item.qspaceH5 as qspaceH5:
            self.__setPlotData(qspaceH5.sample_x,
                               qspaceH5.sample_y,
                               qspaceH5.qspace_sum)
        self.__node = node

        self.__view3d = view3d = ScalarFieldView(self)
        sfDock = Qt.QDockWidget()
        sfTree = SFTreeView()
        sfTree.setSfView(view3d)
        sfDock.setWidget(sfTree)
        view3d.addDockWidget(Qt.Qt.RightDockWidgetArea, sfDock)
        self.setCentralWidget(view3d)
        self.__isoPosition = None

        treeDock = Qt.QDockWidget(self)
        tree = TreeView(self, model=model)
        index = node.index()
        tree.setRootIndex(index)
        # tree.sigCurrentChanged.connect(self.__itemSelected)
        treeDock.setWidget(tree)
        features = treeDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        treeDock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, treeDock)

        plotDock = Qt.QDockWidget(self)
        # tree.sigCurrentChanged.connect(self.__itemSelected)
        plotDock.setWidget(plotWindow)
        features = plotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        plotDock.setFeatures(features)
        # self.addDockWidget(Qt.Qt.RightDockWidgetArea, plotDock)
        self.splitDockWidget(treeDock, plotDock, Qt.Qt.Vertical)

        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        toolBar = self.addToolBar('Fit')
        action = Qt.QAction(icon, 'Fit', toolBar)
        action.setIcon(icon)
        action.triggered.connect(self.__roiApplied)
        # fitBn = Qt.QToolButton()
        # fitBn.setIcon(icon)
        toolBar.addAction(action)

    # TODO : refactor this in a common base with RealSpaceViewWidget
    def __setPlotData(self, x, y, data):
        plot = self.__plotWindow
        # scatter
        min_, max_ = data.min(), data.max()
        colormap = cm.jet
        colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
        plot.addCurve(x, y,
                      color=colors,
                      symbol='s',
                      linestyle='')

    def __roiApplied(self):
        self.sigProcessApplied.emit(self.__node)

    def __plotSignal(self, event):
        if event['event'] not in ('curveClicked',): # , 'mouseClicked'):
            return
        x, y = event['xdata'], event['ydata']

        self.__showIsoView(x, y)

    def __showIsoView(self, x, y):
        isoView = self.__view3d
        plot = self.__plotWindow
        item = h5NodeToProjectItem(self.__node)

        with item.qspaceH5 as qspaceH5:
            sampleX = qspaceH5.sample_x
            sampleY = qspaceH5.sample_y

            # TODO : better
            try:
                xIdx = (np.abs(sampleX - x) + np.abs(sampleY - y)).argmin()
            except:
                xIdx = (np.abs(sampleX - x[0]) + np.abs(sampleY - y[0])).argmin()

            x = sampleX[xIdx]
            y = sampleY[xIdx]

            plot.addXMarker(x, legend='Xselection')
            plot.addYMarker(y, legend='Yselection')

            qspace = qspaceH5.qspace_slice(xIdx)
            isoView.setData(qspace)
