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
from silx.gui import qt as Qt


_registeredNodes = {}


class ModelRoles(object):
    IsXsocsNodeRole, XsocsNodeType, XsocsProcessId, RoleMax = \
        range(Qt.Qt.UserRole, Qt.Qt.UserRole + 4)


class ModelColumns(object):
    NameColumn, ValueColumn, ColumnMax = range(3)


def getNodeClass(nodeType):
    return _registeredNodes.get(nodeType)


def registerNodeClass(klass):
    global _registeredNodes

    nodeType = klass.nodeType
    if nodeType in _registeredNodes:
        raise AttributeError('Failed to register node type {0}.'
                             'Already registered.'
                             ''.format(klass.__name__))

    # TODO : some kind of checks on the klass
    _registeredNodes[nodeType] = klass


def NodeClassDef(nodeType, icon=None, editor=None):
    def inner(cls):
        cls.nodeType = nodeType
        cls.editor = editor
        cls.icon = icon
        registerNodeClass(cls)
        return cls

    return inner


def nodeFactory(projectFile, path, parent=None, nodeType=None):
    klass = None
    if nodeType is not None:
        klass = getNodeClass(nodeType)
        if klass is None:
            raise ValueError('Unknown class type {0}.'.format(nodeType))
    else:
        with h5py.File(projectFile) as h5f:
            item = h5f[path]
            xsocsType = item.attrs.get('XsocsType')
            itemClass = h5f.get(path, getclass=True)
            itemLink = h5f.get(path, getclass=True, getlink=True)
            del item
        if xsocsType is not None:
            klass = getNodeClass(xsocsType)
        if klass is None:
            klass = getNodeClass(itemLink)
        if klass is None:
            klass = getNodeClass(itemClass)
        if klass is None:
            klass = getNodeClass('default')
    if klass is not None:
        return klass(projectFile, path, parent)
    else:
        raise ValueError('Node creation failed.')
