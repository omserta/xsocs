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
from silx.gui.hdf5 import Hdf5TreeModel
from silx.gui import icons

from . import getItemClass


def itemTypeFromIndex(index):
    obj = index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
    xsocsType = obj.attrs.get('XsocsType')
    return xsocsType


def itemClassFromIndex(index):
    xsocsType = itemTypeFromIndex(index)
    xsocsClass = getItemClass(xsocsType)
    return xsocsClass


def itemEditorFromIndex(index):
    xsocsClass = itemClassFromIndex(index)
    if xsocsClass:
        return xsocsClass.editor
    return None


class ProjectModel(Qt.QSortFilterProxyModel):
    TypeRole, IsXsocsNode, TypeLast = \
        range(Hdf5TreeModel.H5PY_OBJECT_ROLE + 1,
              Hdf5TreeModel.H5PY_OBJECT_ROLE + 4)

    def __init__(self, parent=None):
        super(ProjectModel, self).__init__(parent)

    def columnCount(self, parent=Qt.QModelIndex(), *args, **kwargs):
        return 2

    def hasChildren(self, parent=Qt.QModelIndex(), *args, **kwargs):
        obj = parent.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
        xsocsType = obj.attrs.get('XsocsType')
        if xsocsType:
            klass = getItemClass(xsocsType)
            return klass.viewShowChildren
        return super(ProjectModel, self).hasChildren(parent)

    # def flags(self, index):
    #     flags = super(ProjectModel, self).flags(index)
    #     if _xsocsNodeTypeFromIndex is not None:
    #         return flags | Qt.Qt.ItemIsEditable

    def data(self, index, role=Qt.Qt.DisplayRole):
        """
        Reimplemtation of the QSortFilterProxyModel.
        Adds the following roles : XsocsTypeRole, XsocsIsXsocsNode.
        Filters the DecorationRole for XsocsProjectItems
        :param index:
        :param role:
        :return:
        """
        if role == ProjectModel.TypeRole:
            obj = index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
            xsocsType = obj.attrs.get('XsocsType')
            return xsocsType

        if role == ProjectModel.IsXsocsNode:
            obj = index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
            return ((obj.attrs.get('XsocsType') is not None) and 1) or 0

        if role == Qt.Qt.BackgroundRole and \
                index.column() == 0 and \
                itemTypeFromIndex(index) is not None:
            return Qt.QBrush(Qt.QColor(170, 255, 255))

        if (index.column() == Hdf5TreeModel.NAME_COLUMN and
                role == Qt.Qt.DecorationRole):
            obj = index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
            xsocsType = obj.attrs.get('XsocsType')
            if xsocsType:
                klass = getItemClass(xsocsType)
                if klass:
                    return icons.getQIcon(klass.icon)

        return super(ProjectModel, self).data(index, role)

    def filterAcceptsColumn(self, source_column, source_parent):
        if source_column in (Hdf5TreeModel.VALUE_COLUMN,
                             Hdf5TreeModel.NAME_COLUMN):
            return True
        return False
