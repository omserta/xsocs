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

import h5py

_registeredItems = {}


def getItemClass(itemName):
    return _registeredItems.get(itemName)


def registerItemClass(klass):
    global _registeredItems

    itemName = klass.itemName
    if itemName in _registeredItems:
        raise AttributeError('Failed to register item class {0}.'
                             'attribute is already registered.'
                             ''.format(klass.__name__))

    # TODO : some kind of checks on the klass
    _registeredItems[itemName] = klass


def ItemClassDef(itemName, editor=None):
    def inner(cls):
        cls.itemName = itemName
        cls.editor = editor
        registerItemClass(cls)
        return cls
    return inner


class ProjectItem(object):
    itemName = None
    editor = None
    viewShowChildren = True
    icon = None

    XsocsNone, XsocsInput, XsocsQSpace, XsocsFit = range(4)

    def __init__(self, h5File, nodePath, processLevel=None):
        # TODO : check if parent already has a child with the same name
        super(ProjectItem, self).__init__()
        self.__nodePath = nodePath
        self.__h5File = h5File
        self.__processLevel = processLevel or ProjectItem.XsocsNone

    path = property(lambda self: self.__nodePath)

    file = property(lambda self: self.__h5File)

    dataLevel = property(lambda self: self.__dataLevel)

    def _commit(self):
        with h5py.File(self.file, 'a') as h5f:
            grp = h5f.require_group(self.path)
            grp.attrs.update({'XsocsType': self.itemName})
            grp.attrs.update({'XsocsLevel': self.__processLevel or 'None'})

    @classmethod
    def load(cls, h5File, groupPath):
        with h5py.File(h5File, 'r') as h5f:
            grp = h5f[groupPath]
            xsocsType = grp.attrs.get('XsocsType')
            if xsocsType is None:
                return None
            klass = _registeredItems.get(xsocsType)
            if klass is None:
                return None
        instance = klass(h5File, groupPath)
        return instance


class ItemEvent(object):
    def __init__(self, item, eventType):
        super(ItemEvent, self).__init__()
        self.__item = item
        self.__type = eventType

    def plotData(self):
        raise NotImplementedError('')

    type = property(lambda self: self.__type)

    item = property(lambda self: self.__item)
