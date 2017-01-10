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


if __name__ == '__main__':
    pass
