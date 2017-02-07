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

import numpy as np
from matplotlib import cm

from silx.gui import qt as Qt

from ...io.QSpaceH5 import QSpaceH5
from ...io.FitH5 import FitH5Writer, FitH5QAxis

from ..widgets.Containers import GroupBox
from ..widgets.RoiAxisWidget import RoiAxisWidget
from ..widgets.Input import StyledLineEdit

from ...process.peak_fit import PeakFitter, FitTypes


class Roi3DSelectorWidget(Qt.QWidget):
    """
    Widget displaying three RoiAxisWidgets, one for each axis.
    """

    sigRoiChanged = Qt.Signal(object)
    """ Signal emitted when one of the slider is moved. The new ranges are
        passed to the listener : a dictionary with three SliderState instances,
        one for each axis.
    """

    sigRoiToggled = Qt.Signal(bool)
    """ Signal emitted when the QGroupWidget is toggled on/off. """

    def __init__(self, *args, **kwargs):
        super(Roi3DSelectorWidget, self).__init__(*args, **kwargs)

        self.setContentsMargins(0, 0, 0, 0)

        layout = Qt.QVBoxLayout(self)

        self.__grpBox = grpBox = GroupBox('Roi')
        grpBox.setCheckable(True)
        grpBox.setChecked(False)
        grpBox.toggled.connect(self.sigRoiToggled)
        grpLayout = Qt.QVBoxLayout(grpBox)

        xRoiWid = self.__xRoiWid = RoiAxisWidget('X')
        yRoiWid = self.__yRoiWid = RoiAxisWidget('Y')
        zRoiWid = self.__zRoiWid = RoiAxisWidget('Z')

        grpLayout.addWidget(xRoiWid)
        grpLayout.addWidget(yRoiWid)
        grpLayout.addWidget(zRoiWid)

        xRoiWid.sigSliderMoved.connect(self.__slotSliderMoved)
        yRoiWid.sigSliderMoved.connect(self.__slotSliderMoved)
        zRoiWid.sigSliderMoved.connect(self.__slotSliderMoved)

        layout.addWidget(grpBox)

    def __slotSliderMoved(self, sliderEvt):
        """
        Slot called each time a slider moves.
        :param sliderEvt:
        :return:
        """

        sender = self.sender()
        if sender == self.__xRoiWid:
            xState = sliderEvt
            yState = self.__yRoiWid.slider().getSliderState()
            zState = self.__zRoiWid.slider().getSliderState()
        elif sender == self.__yRoiWid:
            xState = self.__xRoiWid.slider().getSliderState()
            yState = sliderEvt
            zState = self.__zRoiWid.slider().getSliderState()
        elif sender == self.__zRoiWid:
            xState = self.__xRoiWid.slider().getSliderState()
            yState = self.__yRoiWid.slider().getSliderState()
            zState = sliderEvt
        elif sender == self.__grpBox:
            return
        else:
            raise RuntimeError('Unknown sender.')

        self.sigRoiChanged.emit({'x': xState,
                                 'y': yState,
                                 'z': zState})

    def isActive(self):
        return self.__grpBox.isChecked()

    def xSlider(self):
        """
        Returns the RangeSlider for the X axis
        :return:
        """
        return self.__xRoiWid.slider()

    def ySlider(self):
        """
        Returns the RangeSlider for the X axis
        :return:
        """
        return self.__yRoiWid.slider()

    def zSlider(self):
        """
        Returns the RangeSlider for the X axis
        :return:
        """
        return self.__zRoiWid.slider()


class FitWidget(Qt.QWidget):
    """
    Fit process widget.
    :param qspaceFile:
    :param kwargs:
    """

    sigProcessDone = Qt.Signal(object)
    """ Signal emitted when a fit is done. Argument is the name of the file
    containing the results.
    """

    sigProcessStarted = Qt.Signal()
    """ Signal emitted when a fit is started. Argument is the name of the file
    containing the results.
    """

    FitTypes = OrderedDict([('Gaussian', FitTypes.GAUSSIAN),
                            ('Centroid', FitTypes.CENTROID),
                            ('Silx', FitTypes.SILX)])

    __sigFitDone = Qt.Signal()

    __progressDelay = 500

    # TODO : pass the actual roi values
    def __init__(self,
                 qspaceFile,
                 **kwargs):
        super(FitWidget, self).__init__(**kwargs)

        self.__qspaceH5 = qspaceH5 = QSpaceH5(qspaceFile)
        self.__progTimer = None

        self.__outputFile = None

        self.__nPeaks = 1

        layout = Qt.QGridLayout(self)

        self.__roiWidget = roiWidget = Roi3DSelectorWidget()

        layout.addWidget(roiWidget)

        fileLayout = Qt.QHBoxLayout()
        self.__fileEdit = fileEdit = StyledLineEdit(nChar=20, readOnly=True)
        fileLayout.addWidget(Qt.QLabel('File :'))
        fileLayout.addWidget(fileEdit)
        layout.addLayout(fileLayout, 1, 0)

        fitLayout = Qt.QHBoxLayout()
        self.__fitTypeCb = fitTypeCb = Qt.QComboBox()
        fitTypeCb.addItems(list(FitWidget.FitTypes.keys()))
        fitTypeCb.setCurrentIndex(0)
        fitLayout.addWidget(Qt.QLabel('Fit :'))
        fitLayout.addWidget(fitTypeCb)
        fitTypeCb.currentIndexChanged[str].connect(
            self.__slotCurrentTextChanged)

        layout.addLayout(fitLayout, 2, 0, alignment=Qt.Qt.AlignLeft)

        self.__nPeaksSpinBox = spinbox = Qt.QSpinBox()
        spinbox.setMinimum(1)
        spinbox.setMaximum(20)
        spinbox.setValue(self.__nPeaks)
        spinbox.setToolTip('Max. number of expected peaks.')
        spinbox.valueChanged.connect(self.__slotValueChanged)
        fitLayout.addWidget(spinbox)
        fitLayout.addWidget(Qt.QLabel('peak(s)'))

        runLayout = Qt.QHBoxLayout()
        self.__runButton = runButton = Qt.QPushButton('Run')
        runButton.setEnabled(False)
        runButton.clicked.connect(self.__slotRunClicked)
        runLayout.addWidget(runButton)

        self.__progBar = progBar = Qt.QProgressBar()
        runLayout.addWidget(progBar)
        layout.addLayout(runLayout, 3, 0, alignment=Qt.Qt.AlignCenter)

        self.__statusLabel = statusLabel = Qt.QLabel('Ready')
        statusLabel.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)
        layout.addWidget(statusLabel, 4, 0)

        with qspaceH5:
            qx = qspaceH5.qx
            qy = qspaceH5.qy
            qz = qspaceH5.qz

        roiWidget.xSlider().setRange([qx[0], qx[-1]])
        roiWidget.ySlider().setRange([qy[0], qy[-1]])
        roiWidget.zSlider().setRange([qz[0], qz[-1]])

        self.__sigFitDone.connect(self.__slotFitDone)

        layout.setRowStretch(layout.rowCount(), 1)
        layout.setColumnStretch(layout.columnCount(), 1)

    def roiWidget(self):
        """
        Returns the Roi3DSelectorWidget instance.
        :return:
        """
        return self.__roiWidget

    def setQSpaceIndex(self, index):
        """
        Selects the qspace cube at *index* in the qspace H5 file, and
        displays the corresponding profiles in the sliders.
        (profile = cube summed along the corresponding axis)
        :param index:
        :return:
        """
        qspace = self.__qspaceH5.qspace_slice(index)
        z_sum = qspace.sum(axis=0).sum(axis=0)
        cube_sum_z = qspace.sum(axis=2)
        y_sum = cube_sum_z.sum(axis=0)
        x_sum = cube_sum_z.sum(axis=1)

        colors = cm.jet(np.arange(255))
        cmap = [Qt.QColor.fromRgbF(*c).rgba() for c in colors]

        roiWidget = self.__roiWidget
        roiWidget.xSlider().setSliderProfile(x_sum, colormap=cmap)
        roiWidget.ySlider().setSliderProfile(y_sum, colormap=cmap)
        roiWidget.zSlider().setSliderProfile(z_sum, colormap=cmap)

    def setOutputFile(self, outputFile):
        self.__outputFile = outputFile
        if outputFile is not None:
            self.__fileEdit.setText(outputFile)
        else:
            self.__fileEdit.clear()
        self.__runButton.setEnabled(outputFile is not None)

    def __slotValueChanged(self, value):
        self.__nPeaks = value

    def __slotCurrentTextChanged(self, text):
        blocked = self.__nPeaksSpinBox.blockSignals(True)
        if text in ('Gaussian', 'Silx'):
            self.__nPeaksSpinBox.setEnabled(True)
            self.__nPeaksSpinBox.setValue(self.__nPeaks)
        else:
            self.__nPeaksSpinBox.setEnabled(False)
            self.__nPeaksSpinBox.setValue(1)
        self.__nPeaksSpinBox.blockSignals(blocked)

    def __slotRunClicked(self):

        # TODO : put some safeguards
        self.__lock(True)

        self.__progBar.setValue(0)
        fitType = FitWidget.FitTypes[self.__fitTypeCb.currentText()]

        if self.__roiWidget.isActive():
            x0, x1 = self.__roiWidget.xSlider().getSliderIndices()
            y0, y1 = self.__roiWidget.ySlider().getSliderIndices()
            z0, z1 = self.__roiWidget.zSlider().getSliderIndices()
            roiIndices = [[x0, x1 + 1], [y0, y1 + 1], [z0, z1 + 1]]
        else:
            roiIndices = None

        self.__fitter = fitter = PeakFitter(self.__qspaceH5.filename,
                                            fit_type=fitType,
                                            roi_indices=roiIndices,
                                            n_peaks=self.__nPeaks)
        self.__statusLabel.setText('Running...')

        self.__progTimer = timer = Qt.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(self.__slotProgTimer)

        try:
            self.sigProcessStarted.emit()
            fitter.peak_fit(blocking=False, callback=self.__sigFitDone.emit)
            timer.start(self.__progressDelay)
        except Exception as ex:
            # TODO : popup
            self.__statusLabel.setText('ERROR')
            print('ERROR : {0}.'.format(ex))
            self.__lock(False)
            self.sigProcessDone.emit(None)

    def __slotProgTimer(self):
        if self.__fitter:
            self.__progBar.setValue(self.__fitter.progress())
        self.__progTimer.start(self.__progressDelay)

    def __lock(self, lock):
        enable = not lock
        self.__roiWidget.setEnabled(enable)
        self.__fitTypeCb.setEnabled(enable)
        self.__runButton.setEnabled(enable)

    def __slotFitDone(self):
        self.__progTimer.stop()
        self.__progTimer = None
        self.__lock(False)

        statusLabel = self.__statusLabel
        fitter = self.__fitter

        status = fitter.status

        try:
            self.__writeResults(fitter.results)
        except Exception as ex:
            # TODO : popup
            print(ex)
            status = PeakFitter.ERROR

        if status == PeakFitter.DONE:
            statusLabel.setText('Succes')
            self.__progBar.setValue(100)
        elif status == PeakFitter.ERROR:
            # TODO : popup
            statusLabel.setText('ERROR')
        elif status == PeakFitter.CANCELED:
            # TODO : popup
            statusLabel.setText('Canceled')
        else:
            # TODO : popup
            statusLabel.setText('?')

        self.__fitter = None

        self.sigProcessDone.emit(self.__outputFile)

    def __writeResults(self, results):
        with FitH5Writer(self.__outputFile, mode='w') as fitH5:

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
                    xresult = results.results(process, param,
                                              results.QX_AXIS)
                    yresult = results.results(process, param,
                                              results.QY_AXIS)
                    zresult = results.results(process, param,
                                              results.QZ_AXIS)

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

            xstatus = results.qx_status()
            ystatus = results.qy_status()
            zstatus = results.qz_status()

            fitH5.set_status(entry,
                             FitH5QAxis.qx_axis,
                             xstatus)
            fitH5.set_status(entry,
                             FitH5QAxis.qy_axis,
                             ystatus)
            fitH5.set_status(entry,
                             FitH5QAxis.qz_axis,
                             zstatus)

if __name__ == '__main__':
    pass