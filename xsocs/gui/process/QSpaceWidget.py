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

from ...io.XsocsH5 import XsocsH5

from ..widgets.Containers import GroupBox
from ..widgets.Input import StyledLineEdit
from ...process.qspace.QSpaceConverter import QSpaceConverter

_ETA_LOWER = u'\u03B7'

_DEFAULT_IMG_BIN = [1, 1]


class ConversionParamsWidget(Qt.QWidget):
    """
    Widget for conversion parameters input :
        - qspace dimensions
        - image binning size
    """
    def __init__(self, **kwargs):
        super(ConversionParamsWidget, self).__init__(**kwargs)
        layout = Qt.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ===========
        # image binning
        # ===========
        row = 0
        layout.addWidget(Qt.QLabel('Img binning :'), row, 0)
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        imgbin_h_edit = StyledLineEdit(nChar=5)
        imgbin_h_edit.setValidator(Qt.QIntValidator(imgbin_h_edit))
        imgbin_h_edit.setAlignment(Qt.Qt.AlignRight)
        imgbin_h_edit.setText(str(_DEFAULT_IMG_BIN[0]))
        h_layout.addWidget(imgbin_h_edit, alignment=Qt.Qt.AlignLeft)
        h_layout.addWidget(Qt.QLabel(' x '))
        imgbin_v_edit = StyledLineEdit(nChar=5)
        imgbin_v_edit.setValidator(Qt.QIntValidator(imgbin_v_edit))
        imgbin_v_edit.setAlignment(Qt.Qt.AlignRight)
        imgbin_v_edit.setText(str(_DEFAULT_IMG_BIN[1]))
        h_layout.addWidget(imgbin_v_edit, alignment=Qt.Qt.AlignLeft)
        h_layout.addWidget(Qt.QLabel('px'))
        h_layout.addStretch(1)

        # ===========
        # qspace size
        # ===========

        row += 1
        layout.addWidget(Qt.QLabel('Q space size :'), row, 0)
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        qsize_x_edit = StyledLineEdit(nChar=5)
        qsize_x_edit.setValidator(Qt.QDoubleValidator(qsize_x_edit))
        qsize_x_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_x_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_y_edit = StyledLineEdit(nChar=5)
        qsize_y_edit.setValidator(Qt.QDoubleValidator(qsize_y_edit))
        qsize_y_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_y_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_z_edit = StyledLineEdit(nChar=5)
        qsize_z_edit.setValidator(Qt.QDoubleValidator(qsize_z_edit))
        qsize_z_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_z_edit)
        h_layout.addWidget(Qt.QLabel('bins'))
        h_layout.addStretch(1)

        self.__imgbin_h_edit = imgbin_h_edit
        self.__imgbin_v_edit = imgbin_v_edit
        self.__qsize_x_edit = qsize_x_edit
        self.__qsize_y_edit = qsize_y_edit
        self.__qsize_z_edit = qsize_z_edit

        # ===========
        # size constraints
        # ===========
        self.setSizePolicy(Qt.QSizePolicy(Qt.QSizePolicy.Fixed,
                                          Qt.QSizePolicy.Fixed))

    @property
    def image_binning(self):
        """
        Returns the image binning, a 2 integers array.
        :return:
        """
        h_bin = self.__imgbin_h_edit.text()
        if len(h_bin) == 0:
            h_bin = None
        else:
            h_bin = int(h_bin)
        v_bin = self.__imgbin_v_edit.text()
        if len(v_bin) == 0:
            v_bin = None
        else:
            v_bin = int(v_bin)
        return [h_bin, v_bin]

    @image_binning.setter
    def image_binning(self, image_binning):
        """
        Sets the image binning.
        :param image_binning: a 2 integers array.
        :return:
        """
        self.__imgbin_h_edit.setText(str(image_binning[0]))
        self.__imgbin_v_edit.setText(str(image_binning[1]))

    @property
    def qspace_size(self):
        """
        Returns the qspace dimensions, a 3 integers (> 1) array if set,
            or [None, None, None].
        :return:
        """
        qsize_x = self.__qsize_x_edit.text()
        if len(qsize_x) == 0:
            qsize_x = None
        else:
            qsize_x = int(qsize_x)
        qsize_y = self.__qsize_y_edit.text()
        if len(qsize_y) == 0:
            qsize_y = None
        else:
            qsize_y = int(qsize_y)
        qsize_z = self.__qsize_z_edit.text()
        if len(qsize_z) == 0:
            qsize_z = None
        else:
            qsize_z = int(qsize_z)
        return [qsize_x, qsize_y, qsize_z]

    @qspace_size.setter
    def qspace_size(self, qspace_size):
        """
        Sets the qspace dimensions.
        :param qspace_size: A three integers array.
        :return:
        """
        self.__qsize_x_edit.setText(str(int(qspace_size[0])))
        self.__qsize_y_edit.setText(str(int(qspace_size[1])))
        self.__qsize_z_edit.setText(str(int(qspace_size[2])))


class QSpaceWidget(Qt.QDialog):
    sigProcessDone = Qt.Signal(object)

    (StatusUnknown, StatusInit,
     StatusRunning, StatusCompleted,
     StatusAborted, StatusCanceled) = StatusList = range(6)

    __sigConvertDone = Qt.Signal()

    def __init__(self,
                 xsocH5File,
                 outQSpaceH5,
                 qspaceDims=None,
                 imageBinning=None,
                 roi=None,
                 entries=None,
                 **kwargs):
        """
        Widgets displaying informations about data to be converted to QSpace,
            and allowing the user to input some parameters.
        :param xsocH5File: name of the input XsocsH5 file.
        :param outQSpaceH5: name of the output hdf5 file
        :param qspaceDims: dimensions of the qspace volume
        :param imageBinning: binning to apply to the images before conversion.
            Default : (1, 1)
        :param roi: Roi in sample coordinates (xMin, xMax, yMin, yMax)
        :param entries: a list of entry names to convert to qspace. If None,
            all entries found in the xsocsH5File will be used.
        :param kwargs:
        """
        super(QSpaceWidget, self).__init__(**kwargs)

        self.__status = QSpaceWidget.StatusInit

        xsocsH5 = XsocsH5(xsocH5File)

        # checking entries
        if entries is None:
            entries = xsocsH5.entries()
        elif len(entries) == 0:
            raise ValueError('At least one entry must be selected.')
        else:
            diff = set(entries) - set(xsocsH5.entries())
            if len(diff) > 0:
                raise ValueError('The following entries were not found in '
                                 'the input file :\n - {0}'
                                 ''.format('\n -'.join(diff)))

        self.__params = {'roi': roi,
                         'xsocsH5_f': xsocH5File,
                         'qspaceH5_f': outQSpaceH5}
        
        topLayout = Qt.QGridLayout(self)

        # ATTENTION : this is done to allow the stretch
        # of the QTableWidget containing the scans info
        topLayout.setColumnStretch(1, 1)

        # ################
        # input QGroupBox
        # ################

        inputGbx = GroupBox("Input")
        layout = Qt.QHBoxLayout(inputGbx)
        topLayout.addWidget(inputGbx,
                            0, 0,
                            1, 2)

        # data HDF5 file input
        lab = Qt.QLabel('XsocsH5 file :')
        xsocsFileEdit = StyledLineEdit(nChar=50, readOnly=True)
        xsocsFileEdit.setText(xsocH5File)
        layout.addWidget(lab,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(xsocsFileEdit,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addStretch()

        # ################
        # Scans
        # ################
        scansGbx = GroupBox("Scans")
        topLayout.addWidget(scansGbx, 1, 1, 2, 1)
        topLayout.setRowStretch(2, 1000)

        grpLayout = Qt.QVBoxLayout(scansGbx)
        infoLayout = Qt.QGridLayout()
        grpLayout.addLayout(infoLayout)

        line = 0
        label = Qt.QLabel('# Roi :')
        self.__roiXMinEdit = xMinText = StyledLineEdit(nChar=5, readOnly=True)
        self.__roiXMaxEdit = xMaxText = StyledLineEdit(nChar=5, readOnly=True)
        self.__roiYMinEdit = yMinText = StyledLineEdit(nChar=5, readOnly=True)
        self.__roiYMaxEdit = yMaxText = StyledLineEdit(nChar=5, readOnly=True)
        roiLayout = Qt.QHBoxLayout()
        roiLayout.addWidget(xMinText)
        roiLayout.addWidget(xMaxText)
        roiLayout.addWidget(yMinText)
        roiLayout.addWidget(yMaxText)
        infoLayout.addWidget(label, line, 0)
        infoLayout.addLayout(roiLayout, line, 1, alignment=Qt.Qt.AlignLeft)

        line += 1
        label = Qt.QLabel('# points :')
        self.__nImgLabel = nImgLabel = StyledLineEdit(nChar=16, readOnly=True)
        nImgLayout = Qt.QHBoxLayout()
        infoLayout.addWidget(label, line, 0)
        infoLayout.addLayout(nImgLayout, line, 1, alignment=Qt.Qt.AlignLeft)
        nImgLayout.addWidget(nImgLabel)
        nImgLayout.addWidget(Qt.QLabel(' (roi / total)'))

        line += 1
        label = Qt.QLabel(u'# {0} :'.format(_ETA_LOWER))
        self.__nAnglesLabel = nAnglesLabel = StyledLineEdit(nChar=5,
                                                            readOnly=True)
        infoLayout.addWidget(label, line, 0)
        infoLayout.addWidget(nAnglesLabel, line, 1, alignment=Qt.Qt.AlignLeft)
        infoLayout.setColumnStretch(2, 1)

        self.__scansTable = scansTable = Qt.QTableWidget(0, 2)
        scansTable.verticalHeader().hide()
        grpLayout.addWidget(scansTable, alignment=Qt.Qt.AlignLeft)

        # ################
        # conversion params
        # ################

        convGbx = GroupBox("Conversion parameters")
        grpLayout = Qt.QVBoxLayout(convGbx)
        topLayout.addWidget(convGbx, 1, 0, alignment=Qt.Qt.AlignTop)

        self.__paramsWid = paramsWid = ConversionParamsWidget()
        grpLayout.addWidget(paramsWid)

        # ################
        # output
        # ################

        outputGbx = GroupBox("Output")
        layout = Qt.QHBoxLayout(outputGbx)
        topLayout.addWidget(outputGbx, 3, 0, 1, 2)
        lab = Qt.QLabel('Output :')
        outputFileEdit = StyledLineEdit(nChar=50, readOnly=True)
        outputFileEdit.setText(outQSpaceH5)
        layout.addWidget(lab,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(outputFileEdit,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addStretch()

        # ################
        # buttons
        # ################

        self.__converBn = convertBn = Qt.QPushButton('Convert')
        cancelBn = Qt.QPushButton('Cancel')
        hLayout = Qt.QHBoxLayout()
        topLayout.addLayout(hLayout, 4, 0, 1, 2,
                            Qt.Qt.AlignHCenter | Qt.Qt.AlignTop)
        hLayout.addWidget(convertBn)
        hLayout.addWidget(cancelBn)

        # #################
        # setting initial state
        # #################

        self.__converter = QSpaceConverter(xsocH5File,
                                           output_f=outQSpaceH5,
                                           qspace_dims=qspaceDims,
                                           img_binning=imageBinning,
                                           roi=roi,
                                           entries=entries)

        cancelBn.clicked.connect(self.close)
        convertBn.clicked.connect(self.__slotConvertBnClicked)

        self.__fillScansInfos()

    def __slotConvertBnClicked(self):
        """
        Slot called when the convert button is clicked. Does some checks
        then starts the conversion if all is OK.
        :return:
        """
        converter = self.__converter
        if converter is None:
            # shouldn't be here
            raise RuntimeError('Converter not found.')
        elif converter.is_running():
            # this part shouldn't even be called, just putting this
            # in case someone decides to modify the code to enable the
            # convert_bn even tho conditions are not met.
            Qt.QMessageBox.critical(self, 'Error',
                                    'A conversion is already in progress!')
            return

        output_file = converter.output_f

        if len(output_file) == 0:
            Qt.QMessageBox.critical(self, 'Error',
                                    'Output file field is mandatory.')
            return

        image_binning = self.__paramsWid.image_binning
        qspace_size = self.__paramsWid.qspace_size

        try:
            converter.image_binning = image_binning
            converter.qspace_dims = qspace_size
        except ValueError as ex:
            Qt.QMessageBox.critical(self, 'Error',
                                    str(ex))
            return

        errors = converter.check_parameters()
        if errors:
            msg = 'Invalid parameters.\n{0}'.format('\n'.join(errors))
            Qt.QMessageBox.critical(self, 'Error', msg)
            return

        if len(converter.check_overwrite()):
            ans = Qt.QMessageBox.warning(self,
                                         'Overwrite?',
                                         ('The output file already exists.'
                                          '\nDo you want to overwrite it?'),
                                         buttons=Qt.QMessageBox.Yes |
                                         Qt.QMessageBox.No)
            if ans == Qt.QMessageBox.No:
                return

        self.__converter = converter
        procDialog = _ConversionProcessDialog(converter, parent=self)
        procDialog.accepted.connect(self.__slotConvertDone)
        procDialog.rejected.connect(self.__slotConvertDone)
        self._setStatus(self.StatusRunning)
        rc = procDialog.exec_()

        if rc == Qt.QDialog.Accepted:
            self.__slotConvertDone()
        procDialog.deleteLater()

    def __slotConvertDone(self):
        """
        Method called when the conversion has been completed succesfuly.
        :return:
        """
        converter = self.__converter
        if not converter:
            return

        self.__qspaceH5 = None
        status = converter.status

        if status == QSpaceConverter.DONE:
            self.__qspaceH5 = converter.results
            self._setStatus(self.StatusCompleted)
            self.hide()
            self.sigProcessDone.emit(self.__qspaceH5)
        elif status == QSpaceConverter.CANCELED:
            self._setStatus(self.StatusAborted)
        else:
            self._setStatus(self.StatusUnknown)

    qspaceH5 = property(lambda self: self.__qspaceH5)
    """ Written file (set when the conversion was succesful, None otherwise. """

    status = property(lambda self: self.__status)
    """ Status of the widget. """

    def _setStatus(self, status):
        """
        Sets the status of the widget.
        :param status:
        :return:
        """
        if status not in QSpaceWidget.StatusList:
            raise ValueError('Unknown status value : {0}.'
                             ''.format(status))
        self.__status = status

    def __fillScansInfos(self):
        """
        Fills the QTableWidget with info found in the input file
        """
        converter = self.__converter
        if converter is None:
            return

        scans = converter.scans
        scansTable = self.__scansTable
        scansTable.setRowCount(len(scans))
        for row, scan in enumerate(scans):
            params = converter.scan_params(scan)
            item = Qt.QTableWidgetItem(scan)
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            scansTable.setItem(row, 0, item)
            item = Qt.QTableWidgetItem(str(params['angle']))
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            scansTable.setItem(row, 1, item)

        scansTable.resizeColumnsToContents()
        width = (sum([scansTable.columnWidth(i)
                     for i in range(scansTable.columnCount())]) +
                 scansTable.verticalHeader().width() +
                 20)
        # TODO : the size is wrong when the
        # verticalScrollBar isnt displayed yet
        # scans_table.verticalScrollBar().width())
        size = scansTable.minimumSize()
        size.setWidth(width)
        scansTable.setMinimumSize(size)

        # TODO : warning if the ROI is empty (too small to contain images)
        params = converter.scan_params(scans[0])
        roi = converter.roi
        if roi is None:
            xMin = xMax = yMin = yMax = 'ns'
        else:
            xMin, xMax, yMin, yMax = roi

        self.__roiXMinEdit.setText(str(xMin))
        self.__roiXMaxEdit.setText(str(xMax))
        self.__roiYMinEdit.setText(str(yMin))
        self.__roiYMaxEdit.setText(str(yMax))

        indices = converter.sample_indices
        nImgTxt = '{0} / {1}'.format(len(indices),
                                     params['n_images'])
        self.__nImgLabel.setText(nImgTxt)

        nEntries = len(XsocsH5(self.__params['xsocsH5_f']).entries())
        self.__nAnglesLabel.setText('{0} / {1}'.format(len(scans), nEntries))


class _ConversionProcessDialog(Qt.QDialog):
    __sigConvertDone = Qt.Signal()

    def __init__(self, converter,
                 parent=None,
                 **kwargs):
        """
        Simple widget displaying a progress bar and a info label during the
            conversion process.
        :param converter:
        :param parent:
        :param kwargs:
        """
        super(_ConversionProcessDialog, self).__init__(parent)
        layout = Qt.QVBoxLayout(self)

        progress_bar = Qt.QProgressBar()
        layout.addWidget(progress_bar)
        status_lab = Qt.QLabel('<font color="blue">Conversion '
                               'in progress</font>')
        status_lab.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)
        layout.addWidget(status_lab)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Abort)
        layout.addWidget(bn_box)
        bn_box.accepted.connect(self.accept)
        bn_box.rejected.connect(self.__onAbort)

        self.__sigConvertDone.connect(self.__convertDone)

        self.__bn_box = bn_box
        self.__progress_bar = progress_bar
        self.__status_lab = status_lab
        self.__converter = converter
        self.__aborted = False

        self.__qtimer = Qt.QTimer()
        self.__qtimer.timeout.connect(self.__onProgress)

        converter.convert(blocking=False,
                          overwrite=True,
                          callback=self.__sigConvertDone.emit,
                          **kwargs)

        self.__qtimer.start(1000)

    def __onAbort(self):
        """
        Slot called when the abort button is clicked.
        :return:
        """
        self.__status_lab.setText('<font color="orange">Cancelling...</font>')
        self.__bn_box.button(Qt.QDialogButtonBox.Abort).setEnabled(False)
        self.__converter.abort(wait=False)
        self.__aborted = True

    def __onProgress(self):
        """
        Slot called when the progress timer timeouts.
        :return:
        """
        progress = self.__converter.progress()
        self.__progress_bar.setValue(progress)

    def __convertDone(self):
        """
        Callback called when the conversion is done (whether its successful or
        not).
        :return:
        """
        self.__qtimer.stop()
        self.__qtimer = None
        self.__onProgress()
        abortBn = self.__bn_box.button(Qt.QDialogButtonBox.Abort)

        converter = self.__converter

        if converter.status == QSpaceConverter.CANCELED:
            self.__bn_box.rejected.disconnect(self.__onAbort)
            self.__status_lab.setText('<font color="red">Conversion '
                                      'cancelled.</font>')
            abortBn.setText('Close')
            self.__bn_box.rejected.connect(self.reject)
            abortBn.setEnabled(True)
        elif converter.status == QSpaceConverter.ERROR:
            self.__bn_box.removeButton(abortBn)
            okBn = self.__bn_box.addButton(Qt.QDialogButtonBox.Ok)
            self.__status_lab.setText('<font color="red">Error : {0}.</font>'
                                      ''.format(converter.status_msg))
            okBn.setText('Close')
        else:
            self.__bn_box.removeButton(abortBn)
            okBn = self.__bn_box.addButton(Qt.QDialogButtonBox.Ok)
            self.__status_lab.setText('<font color="green">Conversion '
                                      'done.</font>')
            okBn.setText('Close')

    status = property(lambda self: 0 if self.__aborted else 1)
    """ Status of the process. """


if __name__ == '__main__':
    pass
