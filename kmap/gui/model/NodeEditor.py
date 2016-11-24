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
__date__ = "01/11/2016"


import weakref
from collections import namedtuple

from silx.gui import qt as Qt

from .ModelDef import ModelRoles

from silx.utils.weakref import WeakMethod


EditorInfo = namedtuple('EditorInfo', ['klass', 'persistent'])


class EditorMixin(object):
    """
    To be used with a Qt.QWidget base.
    """
    persistent = False

    node = property(lambda self: self.__node)

    column = property(lambda self: self.__column)

    def __init__(self, parent, option, index):
        super(EditorMixin, self).__init__(parent)
        self.__node = index.data(ModelRoles.InternalDataRole)
        self.__column = index.column()
        self.setAutoFillBackground(True)
        self.__modelCb = None
        self.__viewCb = None

    @classmethod
    def paint(cls, painter, option, index):
        return False

    def notifyModel(self, *args, **kwargs):
        if not self.__modelCb:
            return
        modelCb = self.__modelCb()
        if modelCb:
            modelCb(self, *args, **kwargs)

    def notifyView(self, *args, **kwargs):
        if not self.__viewCb:
            return
        viewCb = self.__viewCb()
        if viewCb:
            viewCb(self, *args, **kwargs)

    def setViewCallback(self, callback):
        self.__viewCb = WeakMethod(callback)

    def setModelCallback(self, callback):
        self.__modelCb = WeakMethod(callback)

    def sizeHint(self):
        return Qt.QSize(0, 0)

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
