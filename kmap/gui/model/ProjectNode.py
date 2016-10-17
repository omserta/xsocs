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
import h5py
from silx.gui import icons
from .ModelDef import nodeFactory, ModelColumns


class ProjectNode(object):
    nodeName = None
    icon = None
    editor = None

    def __init__(self, projectFile, path, parent=None):
        self.__projectFile = projectFile
        self.__path = path
        self.__parent = parent
        self.__children = None
        self.__childCount = None
        self.__data = [{}, {}]
        if self.icon is not None:
            self.setData(ModelColumns.NameColumn,
                         icons.getQIcon(self.icon),
                         role=Qt.Qt.DecorationRole)

        nodeName = path.rstrip('/').split('/')[-1]
        self.setData(ModelColumns.NameColumn,
                     nodeName,
                     role=Qt.Qt.DisplayRole)

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
    def __init__(self, item, eventType, index):
        super(DelegateEvent, self).__init__()
        self.__item = item
        self.__type = eventType
        self.__index = index

    def plotData(self):
        raise NotImplementedError('')

    type = property(lambda self: self.__type)

    item = property(lambda self: self.__item)

    index = property(lambda self: self.__index)
