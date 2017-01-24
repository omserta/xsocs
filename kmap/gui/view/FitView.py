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


class FitView(Qt.QMainWindow):
    sigPointSelected = Qt.Signal(object)

    __sigInitPlots = Qt.Signal()

    def __init__(self,
                 parent,
                 model,
                 node,
                 qspaceNode,
                 **kwargs):
        super(FitView, self).__init__(parent)

        self.__firstShow = True

        self.setWindowTitle('[XSOCS] {0}'.format(node.h5Path))

        item = h5NodeToProjectItem(node)
        fitH5 = self.__fitH5 = item.fitH5
        qspaceItem = h5NodeToProjectItem(qspaceNode)
        self.__qspaceH5 = qspaceItem.qspaceH5

        with fitH5:
            # only one entry per file supposed right now
            # only one process per entry supposed right now
            self.__entry = fitH5.entries()[0]
            self.__process = fitH5.processes(self.__entry)[0]

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
        tree.setRootIndex(self.__model.index(0, 0, tree.rootIndex()))
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

        self.__sigInitPlots.connect(self.__firstInit, Qt.Qt.QueuedConnection)

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
            self.__sigInitPlots.emit()

    def __firstInit(self):
        """
        Called asynchronously the first time the window is shown.
        This allows us to show the window even though it is not completely
        ready yet (so that the user doesn't have to wait for too long
        before seeing some results).
        :return:
        """
        self.__initPlots()
        self.__startModel()

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

        if process == 'gaussian':
            _initLeastSq(self.__plots, fitH5.filename, entry, process)
        elif process == 'centroid':
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
            process = self.__process
            if process == 'gaussian':
                _plotLeastSq(self.__fitPlots, xIdx,
                             fitH5,
                             entry, process,
                             xAcqQX, xAcqQY, xAcqQZ,
                             yAcqQX, yAcqQY, yAcqQZ)

            elif process == 'centroid':
                _plotCentroid(self.__fitPlots, xIdx,
                              fitH5,
                              entry, process,
                              xAcqQX, xAcqQY, xAcqQZ,
                              yAcqQX, yAcqQY, yAcqQZ)
            else:
                # TODO : popup
                raise ValueError('Unknown process {0}.'.format(process))


# TODO : allow users to register plot functions associated with the kind
# of process results that are being displayed
def _plotLeastSq(plots, index, fitH5,
                 entry, process,
                 xAcqQX, xAcqQY, xAcqQZ,
                 yAcqQX, yAcqQY, yAcqQZ):
    """
    Plots the "leastsq" fit results
    :param plots: plot widgets
    :param index: index of the selected point (in the results array)
    :param fitH5: instance of FitH5. This instance may be already opened by
        the caller.
    :param entry: name of the entry to plot
    :param process: name of the process
    :param xData: x axis values of the fitted data
    :param acqX: x axis values of the acquired data
    :param acqY: y axis values of the acquired data
    :return:
    """

    # TODO : put all this in a toolbox, so it can be shared between
    # the plot and the fit functions
    _const_inv_2_pi_ = np.sqrt(2 * np.pi)
    _gauss_fn = lambda p, pos: (
        p[0] * (1. / (_const_inv_2_pi_ * p[2])) *
        np.exp(-0.5 * ((pos - p[1]) / p[2]) ** 2))

    with fitH5:
        xFitQX = fitH5.get_qx(entry)
        xFitQY = fitH5.get_qy(entry)
        xFitQZ = fitH5.get_qz(entry)

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
    fitted = _gauss_fn(params, xFitQX)
    plots[0].addCurve(xFitQX, fitted, legend='QX LSQ gauss. fit')
    plots[0].addCurve(xAcqQX, yAcqQX, legend='measured')
    plots[0].setGraphTitle('QX / LSQ')

    params = [h_y, p_y, w_y]
    fitted = _gauss_fn(params, xFitQY)
    plots[1].addCurve(xFitQY, fitted, legend='QY LSQ gauss. fit')
    plots[1].addCurve(xAcqQY, yAcqQY, legend='measured')
    plots[1].setGraphTitle('QY / LSQ')

    params = [h_z, p_z, w_z]
    fitted = _gauss_fn(params, xFitQZ)
    plots[2].addCurve(xFitQZ, fitted, legend='QZ LSQ gauss. fit')
    plots[2].addCurve(xAcqQZ, yAcqQZ, legend='measured')
    plots[2].setGraphTitle('QZ / LSQ')


def _plotCentroid(plots, index, fitH5,
                  entry, process,
                  xAcqQX, xAcqQY, xAcqQZ,
                  yAcqQX, yAcqQY, yAcqQZ):
    """
    Plot the results from a "centroid" fit.
    :param plots:
    :param index:
    :param fitH5:
    :param entry:
    :param process:
    :param xAcqQX:
    :param xAcqQY:
    :param xAcqQZ:
    :param yAcqQX:
    :param yAcqQY:
    :param yAcqQZ:
    :return:
    """

    # TODO : put all this in a toolbox, so it can be shared between
    # the plot and the fit functions

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


def _initLeastSq(plots, fitH5Name, entry, process):
    """
    Sets up the plots when the interface is shown for the first time.
    :param plots:
    :param fitH5Name:
    :param entry:
    :param process:
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
    :param plots:
    :param fitH5Name:
    :param entry:
    :param process:
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
