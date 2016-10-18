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


import weakref

import h5py
from silx.gui import icons
from silx.gui import qt as Qt
from .ModelDef import nodeFactory, ModelColumns, ModelRoles


class ProjectNode(object):
    nodeType = None
    icon = None
    editor = None

    def __init__(self, projectFile, path, parent=None):
        self.__projectFile = projectFile
        self.__path = path
        self.__parent = parent
        self.__children = None
        self.__childCount = None
        self.__data = [{}, {}]
        icon = self.icon
        if icon is not None:
            if isinstance(icon, str):
                self.setData(ModelColumns.NameColumn,
                             icons.getQIcon(self.icon),
                             role=Qt.Qt.DecorationRole)
            elif isinstance(icon, Qt.QStyle.StandardPixmap):
                style = Qt.QApplication.style()
                self.setData(ModelColumns.NameColumn,
                             style.standardIcon(icon),
                             role=Qt.Qt.DecorationRole)

        nodeName = path.rstrip('/').split('/')[-1]
        self.setData(ModelColumns.NameColumn,
                     nodeName,
                     role=Qt.Qt.DisplayRole)
        for colIdx in range(ModelColumns.ColumnMax):
            self.setData(colIdx,
                         data=self.nodeType,
                         role=ModelRoles.XsocsNodeType)
            self.setData(colIdx,
                         data=weakref.proxy(self),
                         role=ModelRoles.InternalDataRole)

    def parent(self):
        return self.__parent

    def data(self, column, role=Qt.Qt.DisplayRole):
        if column < 0 or column > len(self.__data):
            return None
        return self.__data[column].get(role)

    def setData(self, column, data, role=Qt.Qt.DisplayRole):
        # WARNING, stores a ref to the data!
        # TODO : check data type
        if column < 0 or column > len(self.__data):
            return
        self.__data[column][role] = data

    def _children(self):
        if self.__children is not None:
            return self.__children

        # TODO : find a better way to do this
        # because it wont work for a dataset
        self.__children = children = []
        with h5py.File(self.__projectFile) as h5f:
            # not using value.name in case this item is an external
            # link : value.name is relative to the external file's root.
            paths = [self.__path + '/' + key
                     for key in h5f[self.__path].keys()]
        newChildren = [nodeFactory(self.__projectFile, path, parent=self)
                       for path in paths]
        children.extend(newChildren)
        self.__childCount = len(children)

        return children

    def row(self):
        if self.__parent:
            # noinspection PyProtectedMember
            return self.__parent._children().index(self)

    def childCount(self):
        if self.__childCount is None:
            with h5py.File(self.__projectFile) as h5f:
                self.__childCount = len(h5f[self.path].keys())
        return self.__childCount

    def hasChildren(self):
        return len(self._children()) > 0

    def child(self, index):
        children = self._children()
        if index < 0 or index >= len(children):
            return None
        return children[index]

    projectFile = property(lambda self: self.__projectFile)
    path = property(lambda self: self.__path)


class DelegateEvent(object):
    def __init__(self, index, data=None):
        super(DelegateEvent, self).__init__()
        self.__data = data
        self.__index = index

    data = property(lambda self: self.__data)

    index = property(lambda self: self.__index)


class NodeDelegate(Qt.QWidget):
    sigEditorEvent = Qt.Signal(object)

    # noinspection PyUnusedLocal
    def __init__(self, parent, option, index):
        super(NodeDelegate, self).__init__(parent)
        self.__index = Qt.QPersistentModelIndex(index)

    def sizeHint(self):
        return Qt.QSize(0, 0)

    def emit(self, event):
        self.sigEditorEvent.emit(event)

    index = property(lambda self: self.__index)
