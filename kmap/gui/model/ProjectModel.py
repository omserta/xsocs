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


from silx.gui import qt as Qt

from .ModelDef import nodeFactory, ModelColumns


def nodeEditorFromIndex(index):
    xsocsClass = index.internalPointer().__class__
    if xsocsClass:
        return xsocsClass.editor
    return None


class ProjectModel(Qt.QAbstractItemModel):
    nameColumn, valueColumn = range(2)

    def __init__(self, projectFile, parent=None):
        super(ProjectModel, self).__init__(parent)
        self.__root = nodeFactory(projectFile, '/', nodeType='RootNode')

    # noinspection PyMethodOverriding
    def parent(self, index):
        if not index.isValid():
            return Qt.QModelIndex()
        node = index.internalPointer()

        if node is None:
            return Qt.QModelIndex()

        parent = node.parent()

        if parent == self.__root:
            return Qt.QModelIndex()

        return self.createIndex(parent.row(), 0, parent)

    def columnCount(self, parent=Qt.QModelIndex(), **kwargs):
        return ModelColumns.ColumnMax

    def headerData(self, section, orientation, role=Qt.Qt.DisplayRole):
        if role == Qt.Qt.DisplayRole and orientation == Qt.Qt.Horizontal:
            if section == 0:
                return 'Name'
            if section == 1:
                return 'Value'
        return None

    def data(self, index, role=Qt.Qt.DisplayRole, **kwargs):
        if not index.isValid():
            raise ValueError('Invalid index.')

        node = index.internalPointer()
        if node is not None:
            return node.data(index.column(), role)

    def index(self, row, column, parent=Qt.QModelIndex(), **kwargs):
        if not self.hasIndex(row, column, parent):
            return Qt.QModelIndex()
        if not parent.isValid():
            node = self.__root
        else:
            node = parent.internalPointer()

        child = node.child(row)

        if child is not None:
            return self.createIndex(row, column, child)
        return Qt.QModelIndex()

    def rowCount(self, parent=Qt.QModelIndex(), **kwargs):
        if not parent.isValid():
            node = self.__root
        else:
            node = parent.internalPointer()
        if node is not None:
            return node.childCount()
        return 0
