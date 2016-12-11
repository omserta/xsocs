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
from silx.gui.plot import PlotWindow, PlotWidget

from matplotlib import cm

from kmap.gui.model.Model import Model, RootNode
from kmap.gui.project.Hdf5Nodes import H5File
from kmap.gui.model.ModelDef import ModelRoles, ModelColumns
from kmap.gui.project.XsocsH5Factory import h5NodeToProjectItem
from kmap.gui.widgets.Containers import GroupBox

from kmap.io.FitH5 import FitH5

from ..widgets.XsocsPlot2D import XsocsPlot2D
from kmap.gui.model.TreeView import TreeView
from kmap.gui.model.NodeEditor import EditorMixin
from kmap.gui.project.Hdf5Nodes import H5Base, H5NodeClassDef


@H5NodeClassDef('FitH5')
class FitH5Node(H5File):
    # TODO : check the file format (make sure that all required
    # groups/datasets are there)

    def _loadChildren(self):
        base = self.h5Path.rstrip('/')
        children = []
        with FitH5(self.h5File, mode='r') as h5f:
            entries = h5f.entries()

        for entry in entries:
            child = FitEntryNode(self.h5File, base + '/' + entry)
            children.append(child)

        return children


@H5NodeClassDef('FitEntry')
class FitEntryNode(H5Base):

    entry = property(lambda self: self.h5Path.lstrip('/').split('/')[0])

    def _loadChildren(self):
        base = self.h5Path.rstrip('/')
        entry = self.entry
        children = []

        with FitH5(self.h5File, mode='r') as h5f:
            processes = h5f.processes(entry)
        for process in processes:
            child = FitProcessNode(self.h5File, base + '/' + process)
            children.append(child)

        return children


class FitProcessNode(FitEntryNode):
    process = property(lambda self: self.h5Path.split('/')[1])

    def _loadChildren(self):
        base = self.h5Path.rstrip('/')
        entry = self.entry
        process = self.process
        children = []

        with FitH5(self.h5File, mode='r') as h5f:
            results = h5f.results(entry, process)
        for result in results:
            child = FitResultNode(self.h5File, base + '/' + result)
            children.append(child)

        return children


class FitResultNode(FitProcessNode):
    result = property(lambda self: self.h5Path.split('/')[-1])

    def __init__(self, *args, **kwargs):
        self.dragEnabledColumns = [False, True, True, True]
        super(FitResultNode, self).__init__(*args, **kwargs)

    def _setupNode(self):
        self.setData(1, 'Qx', Qt.Qt.DisplayRole)
        self.setData(1, Qt.Qt.AlignCenter, Qt.Qt.TextAlignmentRole)
        self.setData(2, 'Qy', Qt.Qt.DisplayRole)
        self.setData(2, Qt.Qt.AlignCenter, Qt.Qt.TextAlignmentRole)
        self.setData(3, 'Qz', Qt.Qt.DisplayRole)
        self.setData(3, Qt.Qt.AlignCenter, Qt.Qt.TextAlignmentRole)

    def _loadChildren(self):
        return [FitThumbnailNode(self.h5File, self.h5Path)]


class FitResultEditor(EditorMixin, PlotWidget):
    persistent = True

    def __init__(self, *args, **kwargs):
        super(FitResultEditor, self).__init__(*args, **kwargs)
        self._backend._enableAxis('left', False)
        self._backend._enableAxis('right', False)
        self._backend.ax.get_xaxis().set_visible(False)
        self._backend.ax.set_xmargin(0)
        self._backend.ax.set_ymargin(0)
        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)
        self.setDataMargins(0, 0, 0, 0)

    def setEditorData(self, index):
        node = index.data(ModelRoles.InternalDataRole)

        if node and not node.setEditorData(self, index.column()):
            value = index.data(Qt.Qt.EditRole)
            return self.setModelValue(value)

        return True

    def setModelValue(self, value):
        return False

    def getEditorData(self):
        pass

    def sizeHint(self):
        return Qt.QSize(100, 100)

    def minimumSizeHint(self):
        return Qt.QSize(100, 100)

    def maximumSize(self):
        return Qt.QSize(100, 100)


class FitThumbnailNode(FitResultNode):
    editors = [None, FitResultEditor, FitResultEditor, FitResultEditor]

    def __init__(self, *args, **kwargs):
        self.dragEnabledColumns = [False, True, True, True]
        super(FitThumbnailNode, self).__init__(*args, **kwargs)

    def _setupNode(self):
        self.setData(ModelColumns.NameColumn, None, Qt.Qt.DisplayRole)

    def _loadChildren(self):
        return []

    def sizeHint(self, column):
        if column == ModelColumns.ValueColumn:
            return Qt.QSize(100, 100)
        return super(FitResultNode, self).sizeHint(column)

    def setEditorData(self, editor, column):
        if not isinstance(editor, FitResultEditor):
            return False
        with FitH5(self.h5File) as fitH5:
            x = fitH5.scan_x(self.entry)
            y = fitH5.scan_y(self.entry)
            if column == 1:
                data = fitH5.get_qx_result(self.entry,
                                           self.process,
                                           self.result)
            elif column == 2:
                data = fitH5.get_qy_result(self.entry,
                                           self.process,
                                           self.result)
            else:
                data = fitH5.get_qz_result(self.entry,
                                           self.process,
                                           self.result)
        min_, max_ = data.min(), data.max()
        colormap = cm.jet
        colors = colormap(
            (data.astype(np.float64) - min_) / (max_ - min_))
        editor.addCurve(x,
                        y,
                        color=colors,
                        symbol='s',
                        linestyle='')
        return True

    def _setModelData(self, editor, column):
        """
        This is called by the View's delegate just before the editor is closed,
        it allows this item to update itself with data from the editor.

        :param editor:
        :return:
        """
        return False

    def _openedEditorEvent(self, editor, column, args=None, kwargs=None):
        """
        This is called by custom editors while they're opened in the view.
        See ItemDelegate.__notifyView. Defqult implementation calls
        _setModelData on this node.

        :param editor:
        :param column:
        :param args: event's args
        :param kwargs: event's kwargs
        :return:
        """

        return self._setModelData(editor, column)


class FitRootNode(RootNode):
    ColumnNames = ['Param', 'Qx', 'Qy', 'Qz']


class FitModel(Model):
    RootNode = FitRootNode
    ColumnsWithDelegates = [1, 2, 3]

    def mimeData(self, indexes):
        if len(indexes) > 1:
            raise ValueError('Drag&Drop of more than one item is not'
                             'supported yet.')
        index = indexes[0]
        node = index.data(ModelRoles.InternalDataRole)

        if not isinstance(node, FitResultNode):
            return super(Model, self).mimeData(indexes)

        if index.column() == 1:
            q_axis = FitH5.qx_axis
        elif index.column() == 2:
            q_axis = FitH5.qy_axis
        elif index.column() == 3:
            q_axis = FitH5.qz_axis
        else:
            raise ValueError('Unexpected column.')

        h5file = node.h5File
        entry = node.entry
        process = node.process
        result = node.result

        data = Qt.QByteArray()
        stream = Qt.QDataStream(data, Qt.QIODevice.WriteOnly)
        stream.writeString(h5file)
        stream.writeString(entry)
        stream.writeString(process)
        stream.writeString(result)
        stream.writeInt(q_axis)

        mimeData = Qt.QMimeData()
        mimeData.setData('application/FitModel', data)

        return mimeData


class DropPlotWidget(XsocsPlot2D):
    sigSelected = Qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super(DropPlotWidget, self).__init__(*args, **kwargs)

        self.__legend = None

        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)
        self.setAcceptDrops(True)
        self.sigPlotSignal.connect(self.__plotSignal)

    def __plotSignal(self, event):
        if event['event'] not in ('mouseClicked',):
            return
        if self.__legend is None:
            return

        x, y = event['x'], event['y']
        self.sigSelected.emit((x, y))

    def dropEvent(self, event):
        mimeData = event.mimeData()
        if not mimeData.hasFormat('application/FitModel'):
            return super(DropPlotWidget, self).dropEvent(event)
        qByteArray = mimeData.data('application/FitModel')
        stream = Qt.QDataStream(qByteArray, Qt.QIODevice.ReadOnly)
        h5File = stream.readString()
        entry = stream.readString()
        process = stream.readString()
        result = stream.readString()
        q_axis = stream.readInt()
        self.plotFitResult(h5File, entry, process, result, q_axis)

    def dragEnterEvent(self, event):
        # super(DropWidget, self).dragEnterEvent(event)
        if event.mimeData().hasFormat('application/FitModel'):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        super(DropPlotWidget, self).dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        super(DropPlotWidget, self).dragMoveEvent(event)

    def plotFitResult(self, fitH5Name, entry, process, result, q_axis):
        with FitH5(fitH5Name) as h5f:
            data = h5f.get_axis_result(entry, process, result, q_axis)
            scan_x = h5f.scan_x(entry)
            scan_y = h5f.scan_y(entry)

        self.__legend = self.setPlotData(scan_x, scan_y, data)
        self.setGraphTitle(result + '/' + FitH5.axis_names[q_axis])


class FitView(Qt.QMainWindow):
    sigProcessApplied = Qt.Signal(object)

    def __init__(self,
                 parent,
                 model,
                 node,
                 qspaceNode,
                 **kwargs):
        super(FitView, self).__init__(parent)

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
        self.__model.startModel()
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
        plot.sigSelected.connect(self.__plotSignal)

        plot = DropPlotWidget(grid=False,
                              curveStyle=False,
                              colormap=False,
                              roi=False,
                              mask=False,
                              yInverted=False)
        grpLayout.addWidget(plot)
        self.__plots.append(plot)
        plot.sigSelected.connect(self.__plotSignal)

        plot = DropPlotWidget(grid=False,
                              curveStyle=False,
                              colormap=False,
                              roi=False,
                              mask=False,
                              yInverted=False)
        grpLayout.addWidget(plot)
        self.__plots.append(plot)
        plot.sigSelected.connect(self.__plotSignal)

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

        plot = XsocsPlot2D()
        plot.setKeepDataAspectRatio(False)
        grpLayout.addWidget(plot)
        self.__fitPlots.append(plot)
        plot.setGraphTitle('Qy fit')

        plot = XsocsPlot2D()
        plot.setKeepDataAspectRatio(False)
        grpLayout.addWidget(plot)
        self.__fitPlots.append(plot)
        plot.setGraphTitle('Qz fit')

        layout.addWidget(grpBox, 0, 2)

        # =================================
        # =================================

        self.setCentralWidget(centralWid)

        self.__initPlots()

    def __initPlots(self):

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

        if process == 'LeastSq':
            _initLeastSq(self.__plots, fitH5.filename, entry, process)
        elif process == 'Centroid':
            _initCentroid(self.__plots, fitH5.filename, entry, process)

    def __plotSignal(self, point):
        x, y = point
        self.__plotFitResults(x, y)

    def __plotFitResults(self, x, y):
        # TODO : the values could/should be loaded when the widget is shown for the
        # first time
        with self.__fitH5 as fitH5:
            sampleX = fitH5.scan_x(self.__entry)
            sampleY = fitH5.scan_y(self.__entry)

            xIdx = ((sampleX - x)**2 + (sampleY - y)**2).argmin()

            x = sampleX[xIdx]
            y = sampleY[xIdx]

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
            if process == 'LeastSq':
                _plotLeastSq(self.__fitPlots, xIdx,
                             fitH5,
                             entry, process,
                             xAcqQX, xAcqQY, xAcqQZ,
                             yAcqQX, yAcqQY, yAcqQZ)

            elif process == 'Centroid':
                _plotCentroid(self.__fitPlots, xIdx,
                              fitH5,
                              entry, process,
                              xAcqQX, xAcqQY, xAcqQZ,
                              yAcqQX, yAcqQY, yAcqQZ)
            else:
                # TODO : popup
                raise ValueError('Unknown process {0}.'.format(process))

        self.__setSelectedPosition(x, y)

    def __setSelectedPosition(self, x, y):
        """Set the selected position.

        :param float x:
        :param float y:
        """
        for plot in self.__plots:
            plot.addXMarker(x, legend='Xselection', color='pink')
            plot.addYMarker(y, legend='Yselection', color='pink')


# TODO : allow users to register plot functions associated with the kind
# of process results that are being displayed
def _plotLeastSq(plots, index, fitH5,
                 entry, process,
                 xAcqQX, xAcqQY, xAcqQZ,
                 yAcqQX, yAcqQY, yAcqQZ):
    """

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

        heights = fitH5.get_result(entry, process, 'height')
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

    # TODO : put all this in a toolbox, so it can be shared between
    # the plot and the fit functions

    positions = fitH5.get_result(entry, process, 'position')

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

    :param plots:
    :param fitH5Name:
    :param entry:
    :param process:
    :return:
    """
    # hard coded result name, this isn't satisfactory but I can't think
    # of any other way right now.
    plots[0].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qx_axis)
    plots[1].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qy_axis)
    plots[2].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qz_axis)


def _initCentroid(plots, fitH5Name, entry, process):
    """

    :param plots:
    :param fitH5Name:
    :param entry:
    :param process:
    :return:
    """
    # hard coded result name, this isn't satisfactory but I can't think
    # of any other way right now.
    plots[0].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qx_axis)
    plots[1].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qy_axis)
    plots[2].plotFitResult(fitH5Name, entry, process, 'position', FitH5.qz_axis)


if __name__ == '__main__':
    pass
