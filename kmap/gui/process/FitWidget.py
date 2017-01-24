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

from collections import OrderedDict
from silx.gui import qt as Qt

from ...io.QSpaceH5 import QSpaceH5
from ...io.FitH5 import FitH5Writer, FitH5QAxis
from ..widgets.Containers import GroupBox
from ..widgets.Input import StyledLineEdit
from ..widgets.FileChooser import FileChooser
from ...process.peak_fit import PeakFitter, FitTypes


class RangeWidget(Qt.QWidget):

    def __init__(self, left, right, **kwargs):
        super(RangeWidget, self).__init__(**kwargs)

        layout = Qt.QGridLayout(self)
        leftEdit = StyledLineEdit(nChar=5)
        rightEdit = StyledLineEdit(nChar=5)
        leftEdit.setReadOnly(True)
        rightEdit.setReadOnly(True)

        layout.addWidget(leftEdit, 0, 1)
        layout.addWidget(rightEdit, 0, 2)

        if left is None:
            leftTxt = 'N/A'
        else:
            leftTxt = '{0:6g}'.format(left)
        if right is None:
            rightTxt = 'N/A'
        else:
            rightTxt = '{0:6g}'.format(right)

        leftEdit.setText(leftTxt)
        rightEdit.setText(rightTxt)


class FitWidget(Qt.QDialog):
    sigProcessDone = Qt.Signal(object)

    (StatusUnknown, StatusInit, StatusRunning, StatusCompleted,
     StatusAborted,
     StatusCanceled) = StatusList = range(6)

    FitTypes = OrderedDict([('LEASTSQ', FitTypes.LEASTSQ),
                            ('CENTROID', FitTypes.CENTROID)])

    __sigConvertDone = Qt.Signal()

    # TODO : pass the actual roi values
    def __init__(self,
                 qspaceFile,
                 outputFile=None,
                 roiIndices=None,
                 **kwargs):
        """

        :param qspaceFile:
        :param outputFile:
        :param roi: array of 3 ranges, or None to fit the whole volume.
        :param kwargs:
        """
        super(FitWidget, self).__init__(**kwargs)

        self.__status = FitWidget.StatusInit

        self.__qspaceFile = qspaceFile
        self.__fitType = None
        self.__selectedFile = None
        self.__fitFile = None

        with QSpaceH5(qspaceFile) as qspaceH5:
            qx = self.__qx = qspaceH5.qx
            qy = self.__qy = qspaceH5.qy
            qz = self.__qz = qspaceH5.qz

        if roiIndices is None:
            roiIndices = [[0, qx.shape[0]], [0, qy.shape[0]], [0, qz.shape[0]]]
            self.__roiIndices = None
        else:
            self.__roiIndices = roiIndices

        # TODO : check validity
        xRoi = qx[[roiIndices[0][0], roiIndices[0][1] - 1]]
        yRoi = qy[[roiIndices[1][0], roiIndices[1][1] - 1]]
        zRoi = qz[[roiIndices[2][0], roiIndices[2][1] - 1]]

        layout = Qt.QGridLayout(self)

        xRangeWid = RangeWidget(*xRoi)
        yRangeWid = RangeWidget(*yRoi)
        zRangeWid = RangeWidget(*zRoi)

        roiGroup = GroupBox('Selection')
        roiLayout = Qt.QGridLayout(roiGroup)
        roiLayout.addWidget(Qt.QLabel('X'), 0, 0)
        roiLayout.addWidget(xRangeWid, 0, 1)
        roiLayout.addWidget(Qt.QLabel('Y'), 1, 0)
        roiLayout.addWidget(yRangeWid, 1, 1)
        roiLayout.addWidget(Qt.QLabel('Z'), 2, 0)
        roiLayout.addWidget(zRangeWid, 2, 1)

        layout.addWidget(roiGroup, 0, 0)

        fitTypeLayout = Qt.QHBoxLayout()
        fitTypeCb = Qt.QComboBox()
        fitTypeCb.currentIndexChanged[str].connect(self.__fitTypeChanged)
        fitTypeCb.addItems(FitWidget.FitTypes.keys())
        fitTypeCb.setCurrentIndex(0)
        self.__fitTypeChanged(fitTypeCb.currentText())
        fitTypeLayout.addWidget(Qt.QLabel('Fit :'))
        fitTypeLayout.addWidget(fitTypeCb)
        layout.addLayout(fitTypeLayout, 1, 0, alignment=Qt.Qt.AlignLeft)

        fChooser = FileChooser(fileMode=Qt.QFileDialog.AnyFile)
        fChooser.sigSelectionChanged.connect(self.__fileChanged)
        fChooser.label.setText('Output')
        if outputFile:
            fChooser.lineEdit.setText(outputFile)
        layout.addWidget(fChooser, 2, 0)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Apply')
        bn_box.accepted.connect(self.__onAccept)
        bn_box.rejected.connect(self.reject)
        layout.addWidget(bn_box, 4, 0)

        layout.setRowStretch(3, 1)

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

        fitter = PeakFitter(self.__qspaceFile,
                            fit_type=self.__fitType,
                            roi_indices=self.__roiIndices)

        results = fitter.peak_fit()

        with FitH5Writer(self.__selectedFile, mode='w') as fitH5:
            entry = results.entry
            fitH5.create_entry(entry)

            fitH5.set_scan_x(entry, results.sample_x)
            fitH5.set_scan_y(entry, results.sample_y)

            fitH5.set_qx(entry, results.q_x)
            fitH5.set_qy(entry, results.q_y)
            fitH5.set_qz(entry, results.q_z)

            processes = results.processes()

            for process in processes:
                fitH5.create_process(entry, process)

                for param in results.params(process):
                    xresult = results.results(process, param, results.QX_AXIS)
                    yresult = results.results(process, param, results.QY_AXIS)
                    zresult = results.results(process, param, results.QZ_AXIS)

                    xstatus = results.qx_status(process)
                    ystatus = results.qy_status(process)
                    zstatus = results.qz_status(process)

                    fitH5.set_qx_result(entry,
                                        process,
                                        param,
                                        xresult)

                    fitH5.set_qy_result(entry,
                                        process,
                                        param,
                                        yresult)

                    fitH5.set_qz_result(entry,
                                        process,
                                        param,
                                        zresult)

                    fitH5.set_status(entry,
                                     process,
                                     FitH5QAxis.qx_axis,
                                     xstatus)
                    fitH5.set_status(entry,
                                     process,
                                     FitH5QAxis.qy_axis,
                                     ystatus)
                    fitH5.set_status(entry,
                                     process,
                                     FitH5QAxis.qz_axis,
                                     zstatus)

        # with FitH5Writer(self.__selectedFile, mode='w') as fitH5:
        #     entry = results.fit_name
        #     process = results.fit_name
        #     fitH5.create_entry(entry)
        #     fitH5.create_process(entry, process)
        #
        #     fitH5.set_scan_x(entry, results.sample_x)
        #     fitH5.set_scan_y(entry, results.sample_y)
        #     fitH5.set_status(entry, process, results.status)
        #     fitH5.set_qx(entry, results.q_x)
        #     fitH5.set_qy(entry, results.q_y)
        #     fitH5.set_qz(entry, results.q_z)
        #
        #     for resName, data in results.q_x_results.items():
        #         fitH5.set_qx_result(entry,
        #                             process,
        #                             resName,
        #                             data)
        #     for resName, data in results.q_y_results.items():
        #         fitH5.set_qy_result(entry,
        #                             process,
        #                             resName,
        #                             data)
        #     for resName, data in results.q_z_results.items():
        #         fitH5.set_qz_result(entry,
        #                             process,
        #                             resName,
        #                             data)

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
