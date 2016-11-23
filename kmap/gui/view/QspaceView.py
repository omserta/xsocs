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

from ...io.QSpaceH5 import QSpaceH5
from ..model.ModelDef import ModelRoles
# from ..project.HybridItem import HybridItem
from ..project.QSpaceGroup import QSpaceItem
from ..model.TreeView import TreeView


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
        # self.setCentralWidget(plotWindow)
        self.__isoView = None
        self.__isoPosition = None
        self.__plotType = None

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
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, plotDock)
        # self.splitDockWidget(treeDock, plotDock, Qt.Qt.Vertical)

    def _emitEvent(self, event):
        self.sigProcessApplied.emit(event)

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
            self.__plotType = 'scatter'
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
            self.__plotType = 'image'
        else:
            raise ValueError('data has {0} dimensions, expected 1 or 2.'
                             ''.format(data.ndim))

    def __plotSignal(self, event):
        if event['event'] not in ('curveClicked',): # , 'mouseClicked'):
            return
        x, y = event['xdata'], event['ydata']

        self.__showIsoView(x, y)

    def __showIsoView(self, x, y):
        print 'SHOW'
        # if self.__isoPosition is not None:
        #     if self.__isoPosition[0] == x and self.__isoPosition[1] == y:
        #         return
        # isoView = self.__isoView
        # if isoView is None:
        #     isoView = IsoViewMainWindow(parent=self)
        #     if isinstance(self.parent(), Qt.QMdiSubWindow):
        #         self.parent().mdiArea().addSubWindow(isoView)

        # node = self.index.data(ModelRoles.InternalDataRole)

        # item = HybridItem(node.projectFile, node.path)

        # if item.hasScatter():
        #     xPos, yPos, _ = item.getScatter()
        # elif item.hasImage():
        #     xPos, yPos, _ = item.getImage()
        # else:
        #     return None

        # # TODO : this wont work with images
        # try:
        #     xIdx = (np.abs(xPos - x)).argmin()
        # except:
        #     print x
        #     xIdx = (np.abs(xPos - x[0])).argmin()
        #
        # # TODO : this is not robust at all
        # qspaceNode = node.parent()
        # qspaceItem = QSpaceItem(qspaceNode.projectFile, qspaceNode.path)
        # qspaceH5 = QSpaceH5(qspaceItem.qspaceFile)
        #
        # qspace = qspaceH5.qspace_slice(xIdx)
        #
        # isoView.setData(qspace)
        # self.__isoView = isoView
        # isoView.show()

#
# class IsoViewMainWindow(ScalarFieldView):
#     """Window displaying an isosurface and some controls."""
#
#     def __init__(self, *args, **kwargs):
#         super(IsoViewMainWindow, self).__init__(*args, **kwargs)
#         self.setAttribute(Qt.Qt.WA_DeleteOnClose)
#
#         # Adjust lighting
#         self.plot3D.viewport.light.direction = 0., 0., -1.
#         self.plot3D.viewport.light.shininess = 32
#         self.plot3D.viewport.bgColor = 0.2, 0.2, 0.2, 1.
#
#         # Add controller dock widget
#         self._control = _Controller(viewer=self)
#         self._control.setWindowTitle('Isosurface')
#         self.addDockWidget(Qt.Qt.RightDockWidgetArea, self._control)
#
#     def setData(self, data, copy=True):
#         data = np.asarray(data)
#         self._control.setRange(data.min(), data.max())
#         super(IsoViewMainWindow, self).setData(data, copy)