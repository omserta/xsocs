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

import os
from collections import OrderedDict
from silx.gui import qt as Qt

from ...io.FitH5 import FitH5Writer
from ..widgets.FileChooser import FileChooser
from ...process.peak_fit import peak_fit, FitTypes


class FitWidget(Qt.QDialog):
    sigProcessDone = Qt.Signal(object)

    (StatusUnknown, StatusInit, StatusRunning, StatusCompleted,
     StatusAborted,
     StatusCanceled) = StatusList = range(6)

    FitTypes = OrderedDict([('LEASTSQ', FitTypes.LEASTSQ),
                            ('CENTROID', FitTypes.CENTROID)])

    __sigConvertDone = Qt.Signal()

    def __init__(self,
                 qspaceFile,
                 outputFile=None,
                 **kwargs):
        super(FitWidget, self).__init__(**kwargs)

        self.__status = FitWidget.StatusInit

        self.__qspaceFile = qspaceFile
        self.__fitType = None
        self.__selectedFile = None
        self.__fitFile = None

        layout = Qt.QGridLayout(self)

        fitTypeLayout = Qt.QHBoxLayout()
        fitTypeCb = Qt.QComboBox()
        fitTypeCb.currentIndexChanged[str].connect(self.__fitTypeChanged)
        fitTypeCb.addItems(FitWidget.FitTypes.keys())
        fitTypeCb.setCurrentIndex(0)
        self.__fitTypeChanged(fitTypeCb.currentText())
        fitTypeLayout.addWidget(Qt.QLabel('Fit :'))
        fitTypeLayout.addWidget(fitTypeCb)
        layout.addLayout(fitTypeLayout, 0, 0, alignment=Qt.Qt.AlignLeft)

        fChooser = FileChooser(fileMode=Qt.QFileDialog.AnyFile)
        fChooser.sigSelectionChanged.connect(self.__fileChanged)
        fChooser.label.setText('Output')
        if outputFile:
            fChooser.lineEdit.setText(outputFile)
        layout.addWidget(fChooser, 1, 0)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Apply')
        bn_box.accepted.connect(self.__onAccept)
        bn_box.rejected.connect(self.reject)
        layout.addWidget(bn_box, 3, 0)

        layout.setRowStretch(2, 1)

        self.__bn_box = bn_box
        self.__fileChanged(outputFile)

    def __fileChanged(self, filePath):
        if filePath:
            enab = True
        else:
            enab = False
        self.__selectedFile = filePath
        bn = self.__bn_box.button(Qt.QDialogButtonBox.Ok)
        bn.setEnabled(enab)

    def __fitTypeChanged(self, fitName):
        self.__fitType = FitWidget.FitTypes[fitName]

    def __onAccept(self):
        self.__fitFile = None

        results = peak_fit(self.__qspaceFile,
                           fit_type=self.__fitType,
                           n_proc=1)
        with FitH5Writer(self.__selectedFile, mode='w') as fitH5:
            fitH5.set_x_fit(results.x_height,
                            results.x_center,
                            results.x_width)
            fitH5.set_y_fit(results.y_height,
                            results.y_center,
                            results.y_width)
            fitH5.set_z_fit(results.z_height,
                            results.z_center,
                            results.z_width)
            fitH5.set_scan_positions(results.sample_x, results.sample_y)
            fitH5.set_status(results.status)

        self.__fitFile = self.__selectedFile
        self._setStatus(FitWidget.StatusCompleted)
        self.accept()

    fitFile = property(lambda self: self.__fitFile)

    status = property(lambda self: self.__status)

    def _setStatus(self, status):
        if status not in FitWidget.StatusList:
            raise ValueError('Unknown status value : {0}.'
                             ''.format(status))
        self.__status = status
