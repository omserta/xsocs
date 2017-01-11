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


import numpy as np

from silx.gui import qt as Qt

from kmap.gui.model.Model import Model, RootNode
from kmap.gui.project.Hdf5Nodes import H5File
from kmap.gui.model.ModelDef import ModelRoles

from kmap.io.FitH5 import FitH5

from ...widgets.XsocsPlot2D import XsocsPlot2D
from ...project.Hdf5Nodes import H5Base, H5NodeClassDef


class PlotGrabber(XsocsPlot2D):
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
        return Qt.QPixmap.grabWidget(self)


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

        return children


class FitProcessNode(FitEntryNode):
    """
    Node linked to a process group in a FitH5 file.
    """
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


if __name__ == '__main__':
    pass