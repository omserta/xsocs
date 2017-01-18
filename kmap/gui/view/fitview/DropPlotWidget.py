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
__date__ = "15/09/2016"

from silx.gui import qt as Qt

from kmap.io.FitH5 import FitH5, FitH5QAxis

from ...widgets.XsocsPlot2D import XsocsPlot2D


class DropPlotWidget(XsocsPlot2D):
    sigSelected = Qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super(DropPlotWidget, self).__init__(*args, **kwargs)

        self.__legend = None

        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)
        self.setAcceptDrops(True)
        self.setPointSelectionEnabled(True)
        self.setShowMousePosition(True)

    def dropEvent(self, event):
        mimeData = event.mimeData()
        if not mimeData.hasFormat('application/FitModel'):
            return super(DropPlotWidget, self).dropEvent(event)
        qByteArray = mimeData.data('application/FitModel')
        stream = Qt.QDataStream(qByteArray, Qt.QIODevice.ReadOnly)
        h5File = stream.readString()
        entry = stream.readString()
        process = stream.readString()
        result = stream.readString()
        q_axis = stream.readInt()
        self.plotFitResult(h5File, entry, process, result, q_axis)

    def dragEnterEvent(self, event):
        # super(DropWidget, self).dragEnterEvent(event)
        if event.mimeData().hasFormat('application/FitModel'):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        super(DropPlotWidget, self).dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        super(DropPlotWidget, self).dragMoveEvent(event)

    def plotFitResult(self, fitH5Name, entry, process, result, q_axis):
        with FitH5(fitH5Name) as h5f:
            data = h5f.get_axis_result(entry, process, result, q_axis)
            scan_x = h5f.scan_x(entry)
            scan_y = h5f.scan_y(entry)

        self.__legend = self.setPlotData(scan_x, scan_y, data)
        self.setGraphTitle(result + '/' + FitH5QAxis.axis_names[q_axis])


if __name__ == '__main__':
    pass
