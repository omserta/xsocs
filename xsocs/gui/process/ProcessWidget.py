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

from collections import namedtuple

from silx.gui import qt as Qt


ProcessResult = namedtuple('ProcessResult', ['status', 'data'])


class ProcessWidget(Qt.QMainWindow):

    sigProcessDone = Qt.Signal(object)

    (StatusUnknown, StatusInit, StatusRunning, StatusCompleted, StatusAborted,
     StatusCanceled) = StatusList = range(6)

    def __init__(self, index=None, parent=None, **kwargs):
        super(ProcessWidget, self).__init__(parent, **kwargs)
        self.__index = index
        self.__status = ProcessWidget.StatusInit

    status = property(lambda self: self.__status)

    def _setStatus(self, status):
        if status not in ProcessWidget.StatusList:
            raise ValueError('Unknown status value : {0}.'
                             ''.format(status))
        self.__status = status

    def processResult(self):
        return ProcessResult(status=self.status, data=self._processData())

    def _processData(self):
        return None

    def _emitEvent(self, event):
        self.sigProcessDone.emit(event)

    index = property(lambda self: self.__index)


class ProcessWidgetEvent(object):

    def __init__(self, widget, data=None):
        super(ProcessWidgetEvent, self).__init__()
        self.__index = widget.index
        self.__data = data

    data = property(lambda self: self.__data)

    index = property(lambda self: self.__index)
