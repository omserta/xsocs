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

from silx.gui import qt as Qt
from kmap.gui.project.XsocsH5Factory import h5NodeToProjectItem
from kmap.gui.widgets.Containers import GroupBox

from kmap.io.FitH5 import FitH5, FitH5QAxis

from ..widgets.XsocsPlot2D import XsocsPlot2D
from kmap.gui.model.TreeView import TreeView
from .fitview.FitModel import FitModel, FitH5Node
from .fitview.DropPlotWidget import DropPlotWidget

from.fitview.ResultPlot import plotCentroid, plotGaussian, plotSilx

class FitView(Qt.QMainWindow):
    sigPointSelected = Qt.Signal(object)

    __sigInitPlots = Qt.Signal()

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(FitView, self).__init__(parent)

        self.__firstShow = True

        self.setWindowTitle('[XSOCS] {0}'.format(node.h5Path))

        item = h5NodeToProjectItem(node)
        fitH5 = self.__fitH5 = item.fitH5

        # TODO : this parent().parent() thing is ugly...
        qspaceItem = h5NodeToProjectItem(node.parent().parent())

        self.__qspaceH5 = qspaceItem.qspaceH5
        self.__node = node

        with fitH5:
            # only one entry per file supposed right now
            self.__entry = fitH5.entries()[0]

        centralWid = Qt.QWidget()
        layout = Qt.QGridLayout(centralWid)

        self.__plots = []
        self.__fitPlots = []

        treeDock = Qt.QDockWidget()

        self.__model = FitModel()
        rootNode = FitH5Node(item.fitFile)
        self.__model.appendGroup(rootNode)

        tree = self.__tree = TreeView()
        tree.setModel(self.__model)
        # tree.setRootIndex(self.__model.index(0, 0, tree.rootIndex()))
        tree.setSelectionBehavior(Qt.QAbstractItemView.SelectItems)
        tree.header().setStretchLastSection(False)
        tree.setShowUniqueGroup(True)
        tree.setDragDropMode(Qt.QAbstractItemView.DragDrop)

        treeDock.setWidget(tree)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, treeDock)

        grpBox = GroupBox('Maps')
        grpLayout = Qt.QVBoxLayout(grpBox)

        plot = DropPlotWidget(grid=False,
                              curveStyle=False,
                              colormap=False,
                              roi=False,
                              mask=False,
                              yInverted=False)
        grpLayout.addWidget(plot)
        self.__plots.append(plot)
        plot.sigPointSelected.connect(self.__slotPointSelected)

        plot = DropPlotWidget(grid=False,
                              curveStyle=False,
                              colormap=False,
                              roi=False,
                              mask=False,
                              yInverted=False)
        grpLayout.addWidget(plot)
        self.__plots.append(plot)
        plot.sigPointSelected.connect(self.__slotPointSelected)

        plot = DropPlotWidget(grid=False,
                              curveStyle=False,
                              colormap=False,
                              roi=False,
                              mask=False,
                              yInverted=False)
        grpLayout.addWidget(plot)
        self.__plots.append(plot)
        plot.sigPointSelected.connect(self.__slotPointSelected)

        layout.addWidget(grpBox, 0, 1)

        # =================================
        # =================================
        grpBox = GroupBox('Fit')
        grpLayout = Qt.QVBoxLayout(grpBox)

        plot = XsocsPlot2D()
        plot.setKeepDataAspectRatio(False)
        grpLayout.addWidget(plot)
        self.__fitPlots.append(plot)
        plot.setGraphTitle('Qx fit')
        plot.setShowMousePosition(True)

        plot = XsocsPlot2D()
        plot.setKeepDataAspectRatio(False)
        grpLayout.addWidget(plot)
        self.__fitPlots.append(plot)
        plot.setGraphTitle('Qy fit')
        plot.setShowMousePosition(True)

        plot = XsocsPlot2D()
        plot.setKeepDataAspectRatio(False)
        grpLayout.addWidget(plot)
        self.__fitPlots.append(plot)
        plot.setGraphTitle('Qz fit')
        plot.setShowMousePosition(True)

        layout.addWidget(grpBox, 0, 2)

        # =================================
        # =================================

        self.setCentralWidget(centralWid)

    def getFitNode(self):
        return self.__node

    def showEvent(self, event):
        """
        Overloard method from Qt.QWidget.showEvent to set up the widget the
        first time it is shown. Also starts the model in a queued slot, so that
        the window is shown right away, and the thumbmails are drawn
        afterwards.
        :param event:
        :return:
        """
        # TODO : this is a workaround to the fact that
        # plot ranges aren't set correctly when adding data when the plot
        # widget hasn't been shown yet.
        Qt.QCoreApplication.processEvents()
        super(FitView, self).showEvent(event)
        if self.__firstShow:
            self.__firstShow = False
            self.__firstInit()
            # self.__sigInitPlots.emit()

    def __firstInit(self):
        """
        Called the first time the window is shown.
        :return:
        """
        initDiag = Qt.QProgressDialog('Setting up fit view.', 'cc', 0, 100,
                                      parent=self.parent())
        initDiag.setWindowTitle('Please wait...')
        initDiag.setCancelButton(None)
        initDiag.setAttribute(Qt.Qt.WA_DeleteOnClose)
        initDiag.show()
        initDiag.setValue(10)
        self.__initPlots()
        initDiag.setValue(40)
        self.__startModel()
        initDiag.setValue(70)
        tree = self.__tree
        root = self.__model.index(0, 0, tree.rootIndex())
        tree.setRootIndex(self.__model.index(0, 0, root))
        initDiag.setValue(90)
        tree.expandAll()
        initDiag.setValue(100)
        initDiag.accept()
        initDiag.close()

    def __startModel(self):
        """
        Starts the model (in this case draws the thumbnails
        :return:
        """
        self.__model.startModel()

    def __initPlots(self):
        """
        Initializes the "map" plots.
        :return:
        """
        fitH5 = self.__fitH5

        entry = None
        process = None
        with fitH5:
            entries = fitH5.entries()
            if entries:
                entry = entries[0]
                processes = fitH5.processes(entry)
                if processes:
                    process = processes[0]

        if entry in ('Gaussian', 'SilxFit'):
            _initGaussian(self.__plots, fitH5.filename, entry, process)
        elif process == 'Centroid':
            _initCentroid(self.__plots, fitH5.filename, entry, process)

    def __slotPointSelected(self, point):
        """
        Called when a point is selected on one of the "map" plots.
        :param point:
        :return:
        """
        sender = self.sender()
        for plot in self.__plots:
            if plot != sender:
                plot.selectPoint(point.x, point.y)

        self.__plotFitResults(point.xIdx)
        self.sigPointSelected.emit(point)

    def __plotFitResults(self, xIdx):
        """
        Plots the fit results for the selected point on the plot.
        :param xIdx:
        :return:
        """

        with self.__fitH5 as fitH5:

            entry = self.__entry

            qspaceH5 = self.__qspaceH5
            with qspaceH5:
                cube = qspaceH5.qspace_slice(xIdx)
                histo = qspaceH5.histo
                mask = np.where(histo > 0)
                weights = histo[mask]
                cube[mask] /= weights
                xAcqQX = qspaceH5.qx
                xAcqQY = qspaceH5.qy
                xAcqQZ = qspaceH5.qz

            yAcqQZ = cube.sum(axis=0).sum(axis=0)
            cube_sum_z = cube.sum(axis=2)
            yAcqQY = cube_sum_z.sum(axis=0)
            yAcqQX = cube_sum_z.sum(axis=1)

            for plot in self.__fitPlots:
                plot.clearCurves()
                plot.clearMarkers()

            # TODO : refactor
            if entry == 'Gaussian':
                plotGaussian(self.__fitPlots, xIdx,
                             fitH5, entry,
                             xAcqQX, xAcqQY, xAcqQZ,
                             yAcqQX, yAcqQY, yAcqQZ)

            elif entry == 'Centroid':
                plotCentroid(self.__fitPlots, xIdx,
                             fitH5, entry,
                             xAcqQX, xAcqQY, xAcqQZ,
                             yAcqQX, yAcqQY, yAcqQZ)
            elif entry == 'SilxFit':
                plotSilx(self.__fitPlots, xIdx,
                             fitH5, entry,
                             xAcqQX, xAcqQY, xAcqQZ,
                             yAcqQX, yAcqQY, yAcqQZ)
            else:
                # TODO : popup
                raise ValueError('Unknown entry {0}.'.format(entry))


def _initGaussian(plots, fitH5Name, entry, process):
    """
    Sets up the plots when the interface is shown for the first time.
    :param plots: the plot widgets
    :param fitH5Name: fitH5 file name
    :param entry: name of the entry in the fitH5
    :param process: name of the process in the fitH5
    :return:
    """
    # hard coded result name, this isn't satisfactory but I can't think
    # of any other way right now.

    qApp = Qt.qApp
    qApp.processEvents()

    plots[0].plotFitResult(fitH5Name, entry, process,
                           'position', FitH5QAxis.qx_axis)

    qApp.processEvents()

    plots[1].plotFitResult(fitH5Name, entry, process,
                           'position', FitH5QAxis.qy_axis)

    qApp.processEvents()

    plots[2].plotFitResult(fitH5Name, entry, process,
                           'position', FitH5QAxis.qz_axis)


def _initCentroid(plots, fitH5Name, entry, process):
    """
    Sets up the plots when the interface is shown for the first time.
    :param plots: the plot widgets
    :param fitH5Name: fitH5 file name
    :param entry: name of the entry in the fitH5
    :param process: name of the process in the fitH5
    :return:
    """
    # hard coded result name, this isn't satisfactory but I can't think
    # of any other way right now.
    qApp = Qt.qApp

    qApp.processEvents()
    plots[0].plotFitResult(fitH5Name, entry,
                           process, 'COM', FitH5QAxis.qx_axis)

    qApp.processEvents()
    plots[1].plotFitResult(fitH5Name, entry,
                           process, 'COM', FitH5QAxis.qy_axis)

    qApp.processEvents()
    plots[2].plotFitResult(fitH5Name, entry,
                           process, 'COM', FitH5QAxis.qz_axis)


if __name__ == '__main__':
    pass
