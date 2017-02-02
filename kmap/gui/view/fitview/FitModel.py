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
__date__ = "01/01/2017"

from silx.gui import qt as Qt

import numpy as np

from kmap.gui.model.Model import Model, RootNode
from kmap.gui.project.Hdf5Nodes import H5File
from kmap.gui.model.ModelDef import ModelRoles

from kmap.io.FitH5 import FitH5, FitH5QAxis
from ....process.peak_fit import FitStatus

from ...widgets.XsocsPlot2D import XsocsPlot2D
from ...project.Hdf5Nodes import H5Base, H5NodeClassDef


def _grabWidget(widget):
    """
    Grabs a widget and returns a pixmap.
    :param widget:
    :return:
    """

    if int(Qt.qVersion().split('.')[0]) <= 4:
        pixmap = Qt.QPixmap.grabWidget(widget)
    else:
        pixmap = widget.grab()
    return pixmap


class PlotGrabber(XsocsPlot2D):
    """
    XsocsPlot2D that can be converted to a pixmap.
    """
    persistent = True

    def __init__(self, *args, **kwargs):
        super(PlotGrabber, self).__init__(*args, **kwargs)
        self._backend._enableAxis('left', False)
        self._backend._enableAxis('right', False)
        self._backend.ax.get_xaxis().set_visible(False)
        self._backend.ax.set_xmargin(0)
        self._backend.ax.set_ymargin(0)
        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)
        self.setDataMargins(0, 0, 0, 0)
        self.setCollaspibleMenuVisible(False)
        self.setPointWidgetVisible(False)

    def toPixmap(self):
        """
        Returns a pixmap of the widget.
        :return:
        """
        return _grabWidget(self)




@H5NodeClassDef('FitH5')
class FitH5Node(H5File):
    """
    Node linked to a FitH5 file.
    """
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
    """
    Node linked to an entry in a FitH5 file.
    """

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

        statusNode = FitStatusNode(self.h5File, base)
        children.append(statusNode)

        return children

    def mimeData(self, column, stream):
        # TODO : put column value in enum
        if column == 1:
            q_axis = FitH5QAxis.qx_axis
        elif column == 2:
            q_axis = FitH5QAxis.qy_axis
        elif column == 3:
            q_axis = FitH5QAxis.qz_axis
        else:
            raise ValueError('Unexpected column.')

        h5file = self.h5File
        entry = self.entry

        stream.writeQString(h5file)
        stream.writeQString(entry)
        stream.writeInt(q_axis)

        return True


class FitProcessNode(FitEntryNode):
    """
    Node linked to a process group in a FitH5 file.
    """
    process = property(lambda self: self.h5Path.lstrip('/').split('/')[1])

    def _loadChildren(self):
        base = self.h5Path.rstrip('/')
        entry = self.entry
        process = self.process
        children = []
        with FitH5(self.h5File, mode='r') as h5f:
            results = h5f.get_result_names(entry, process)
        for result in results:
            child = FitResultNode(self.h5File,
                                  base + '/' + result)
            children.append(child)

        return children


class FitStatusNode(FitEntryNode):
    """
    Preview of the points where the fit has failed.
    """
    def __init__(self, *args, **kwargs):
        self.dragEnabledColumns = [False, True, True, True]
        super(FitStatusNode, self).__init__(*args, **kwargs)
        self.nodeName = 'Status'

        self.__nErrors = [0, 0, 0]

    def _setupNode(self):
        width = 100
        plot = PlotGrabber()
        plot.setFixedSize(Qt.QSize(width, 100))
        plot.toPixmap()

        qApp = Qt.qApp
        qApp.processEvents()

        with FitH5(self.h5File) as fitH5:
            x = fitH5.scan_x(self.entry)
            y = fitH5.scan_y(self.entry)

            status = fitH5.get_qx_status(self.entry)
            errorPts = np.where(status != FitStatus.OK)[0]
            self.__nErrors[0] = len(errorPts)
            if len(errorPts) != 0:
                plot.setPlotData(x[errorPts], y[errorPts], status[errorPts])
                pixmap = plot.toPixmap()
            else:
                label = Qt.QLabel('No errors')
                label.setFixedWidth(width)
                label.setAlignment(Qt.Qt.AlignCenter)
                label.setAttribute(Qt.Qt.WA_TranslucentBackground)
                pixmap = _grabWidget(label)
            self.setData(1, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

            status = fitH5.get_qy_status(self.entry)
            errorPts = np.where(status != FitStatus.OK)[0]
            self.__nErrors[1] = len(errorPts)
            if len(errorPts) != 0:
                plot.setPlotData(x[errorPts], y[errorPts], status[errorPts])
                pixmap = plot.toPixmap()
            else:
                label = Qt.QLabel('No errors')
                label.setFixedWidth(width)
                label.setAlignment(Qt.Qt.AlignCenter)
                label.setAttribute(Qt.Qt.WA_TranslucentBackground)
                pixmap = _grabWidget(label)
            self.setData(2, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

            status = fitH5.get_qz_status(self.entry)
            errorPts = np.where(status != FitStatus.OK)[0]
            self.__nErrors[2] = len(errorPts)
            if len(errorPts) != 0:
                plot.setPlotData(x[errorPts], y[errorPts], status[errorPts])
                pixmap = plot.toPixmap()
            else:
                label = Qt.QLabel('No errors')
                label.setFixedWidth(width)
                label.setAlignment(Qt.Qt.AlignCenter)
                label.setAttribute(Qt.Qt.WA_TranslucentBackground)
                pixmap = _grabWidget(label)
            self.setData(3, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

    def _loadChildren(self):
        return []

    def mimeData(self, column, stream):

        if column < 1 or column > 3:
            return False

        if self.__nErrors[column - 1] == 0:
            return False

        if not FitEntryNode.mimeData(self, column, stream):
            return False

        stream.writeQString('status')

        return True


class FitResultNode(FitProcessNode):
    """
    Node linked to a result group in a FitH5 file.
    """
    result = property(lambda self: self.h5Path.split('/')[-1])

    def __init__(self, *args, **kwargs):
        self.dragEnabledColumns = [False, True, True, True]
        super(FitResultNode, self).__init__(*args, **kwargs)

    def _setupNode(self):
        plot = PlotGrabber()
        plot.setFixedSize(Qt.QSize(100, 100))
        plot.toPixmap()

        qApp = Qt.qApp
        qApp.processEvents()

        with FitH5(self.h5File) as fitH5:
            x = fitH5.scan_x(self.entry)
            y = fitH5.scan_y(self.entry)

            data = fitH5.get_qx_result(self.entry,
                                       self.process,
                                       self.result)
            plot.setPlotData(x, y, data)
            pixmap = plot.toPixmap()
            self.setData(1, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

            data = fitH5.get_qy_result(self.entry,
                                       self.process,
                                       self.result)
            plot.setPlotData(x, y, data)
            pixmap = plot.toPixmap()
            self.setData(2, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

            data = fitH5.get_qz_result(self.entry,
                                       self.process,
                                       self.result)
            plot.setPlotData(x, y, data)
            pixmap = plot.toPixmap()
            self.setData(3, pixmap, Qt.Qt.DecorationRole)
            qApp.processEvents()

    def _loadChildren(self):
        return []

    def mimeData(self, column, stream):
        if not FitProcessNode.mimeData(self, column, stream):
            return False

        process = self.process
        result = self.result
        stream.writeQString('result')
        stream.writeQString(process)
        stream.writeQString(result)

        return True


class FitRootNode(RootNode):
    """
    Root node for the FitModel
    """
    ColumnNames = ['Param', 'Qx', 'Qy', 'Qz']


class FitModel(Model):
    """
    Model displaying a FitH5 file contents.
    """
    RootNode = FitRootNode
    ColumnsWithDelegates = [1, 2, 3]

    def mimeData(self, indexes):
        if len(indexes) > 1:
            raise ValueError('Drag&Drop of more than one item is not'
                             'supported yet.')

        mimeData = Qt.QMimeData()

        index = indexes[0]
        node = index.data(ModelRoles.InternalDataRole)

        if not isinstance(node, (FitResultNode, FitStatusNode)):
            return super(Model, self).mimeData(indexes)

        data = Qt.QByteArray()
        stream = Qt.QDataStream(data, Qt.QIODevice.WriteOnly)
        if node.mimeData(index.column(), stream):
            mimeData.setData('application/FitModel', data)

        return mimeData


if __name__ == '__main__':
    pass
