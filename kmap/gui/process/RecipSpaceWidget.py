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

from functools import partial
from collections import namedtuple

from silx.gui import qt as Qt

from ..widgets.Containers import GroupBox
from ..Widgets import (AdjustedLineEdit,
                       AdjustedPushButton)
from ...process.qspace import RecipSpaceConverter

_ETA_LOWER = u'\u03B7'

_DEFAULT_IMG_BIN = [1, 1]


class ConversionParamsWidget(Qt.QWidget):
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
        imgbin_h_edit = AdjustedLineEdit(5)
        imgbin_h_edit.setValidator(Qt.QIntValidator(imgbin_h_edit))
        imgbin_h_edit.setAlignment(Qt.Qt.AlignRight)
        imgbin_h_edit.setText(str(_DEFAULT_IMG_BIN[0]))
        h_layout.addWidget(imgbin_h_edit, alignment=Qt.Qt.AlignLeft)
        h_layout.addWidget(Qt.QLabel(' x '))
        imgbin_v_edit = AdjustedLineEdit(5)
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
        qsize_x_edit = AdjustedLineEdit(5)
        qsize_x_edit.setValidator(Qt.QDoubleValidator(qsize_x_edit))
        qsize_x_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_x_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_y_edit = AdjustedLineEdit(5)
        qsize_y_edit.setValidator(Qt.QDoubleValidator(qsize_y_edit))
        qsize_y_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_y_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_z_edit = AdjustedLineEdit(5)
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
        self.__imgbin_h_edit.setText(str(image_binning[0]))
        self.__imgbin_v_edit.setText(str(image_binning[1]))

    @property
    def qspace_size(self):
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
        self.__qsize_x_edit.setText(str(int(qspace_size[0])))
        self.__qsize_y_edit.setText(str(int(qspace_size[1])))
        self.__qsize_z_edit.setText(str(int(qspace_size[2])))


class RecipSpaceWidget(Qt.QDialog):
    sigProcessDone = Qt.Signal(object)

    (StatusUnknown, StatusInit,
     StatusRunning, StatusCompleted,
     StatusAborted, StatusCanceled) = StatusList = range(6)

    __sigConvertDone = Qt.Signal()

    def __init__(self,
                 data_h5f=None,
                 output_f=None,
                 qspace_size=None,
                 image_binning=None,
                 rect_roi=None,
                 **kwargs):
        super(RecipSpaceWidget, self).__init__(**kwargs)

        self.__status = RecipSpaceWidget.StatusInit
        # self.__central = Qt.QWidget()
        # self.setCentralWidget(self.__central)
        topLayout = Qt.QGridLayout(self)

        self.__rectRoi = rect_roi

        # ATTENTION : this is done to allow the stretch
        # of the QTableWidget containing the scans info
        topLayout.setColumnStretch(1, 1)

        # ################
        # input QGroupBox
        # ################

        input_gbx = GroupBox("Input")
        layout = Qt.QHBoxLayout(input_gbx)
        topLayout.addWidget(input_gbx,
                            0, 0,
                            1, 2)

        # data HDF5 file input
        lab = Qt.QLabel('HDF5 file :')
        h5_file_edit = Qt.QLineEdit()
        fm = h5_file_edit.fontMetrics()
        h5_file_edit.setMinimumWidth(fm.width(' ' * 100))
        h5_file_bn = AdjustedPushButton('...')
        layout.addWidget(lab,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(h5_file_edit,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(h5_file_bn,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addStretch()

        # ################
        # Scans
        # ################
        scans_gbx = GroupBox("Scans")
        topLayout.addWidget(scans_gbx, 1, 1, 2, 1)
        topLayout.setRowStretch(2, 1000)

        grp_layout = Qt.QVBoxLayout(scans_gbx)
        info_layout = Qt.QGridLayout()
        grp_layout.addLayout(info_layout)

        line = 0
        label = Qt.QLabel('# Roi :')
        xMinText = AdjustedLineEdit(width=5, read_only=True)
        xMaxText = AdjustedLineEdit(width=5, read_only=True)
        yMinText = AdjustedLineEdit(width=5, read_only=True)
        yMaxText = AdjustedLineEdit(width=5, read_only=True)
        roi_layout = Qt.QHBoxLayout()
        roi_layout.addWidget(xMinText)
        roi_layout.addWidget(xMaxText)
        roi_layout.addWidget(yMinText)
        roi_layout.addWidget(yMaxText)
        info_layout.addWidget(label, line, 0)
        info_layout.addLayout(roi_layout, line, 1, alignment=Qt.Qt.AlignLeft)

        line += 1
        label = Qt.QLabel('# points :')
        n_img_label = AdjustedLineEdit(width=16, read_only=True)
        nImgLayout = Qt.QHBoxLayout()
        info_layout.addWidget(label, line, 0)
        info_layout.addLayout(nImgLayout, line, 1, alignment=Qt.Qt.AlignLeft)
        nImgLayout.addWidget(n_img_label)
        nImgLayout.addWidget(Qt.QLabel(' (roi / total)'))

        line += 1
        label = Qt.QLabel(u'# {0} :'.format(_ETA_LOWER))
        n_angles_label = AdjustedLineEdit(5, read_only=True)
        info_layout.addWidget(label, line, 0)
        info_layout.addWidget(n_angles_label, line, 1,
                              alignment=Qt.Qt.AlignLeft)
        info_layout.setColumnStretch(2, 1)

        scans_table = Qt.QTableWidget(0, 2)
        scans_table.verticalHeader().hide()
        grp_layout.addWidget(scans_table, alignment=Qt.Qt.AlignLeft)

        # ################
        # conversion params
        # ################

        conv_gbx = GroupBox("Conversion parameters")
        grp_layout = Qt.QVBoxLayout(conv_gbx)
        topLayout.addWidget(conv_gbx, 1, 0, alignment=Qt.Qt.AlignTop)

        conv_params_wid = ConversionParamsWidget()
        grp_layout.addWidget(conv_params_wid)

        # ################
        # output
        # ################

        output_gbx = GroupBox("Output")
        layout = Qt.QHBoxLayout(output_gbx)
        topLayout.addWidget(output_gbx, 3, 0, 1, 2)
        lab = Qt.QLabel('File :')
        output_file_edit = Qt.QLineEdit()
        fm = output_file_edit.fontMetrics()
        output_file_edit.setMinimumWidth(fm.width(' ' * 100))
        output_file_bn = AdjustedPushButton('...')
        layout.addWidget(lab,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(output_file_edit,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addWidget(output_file_bn,
                         stretch=0,
                         alignment=Qt.Qt.AlignLeft)
        layout.addStretch()

        # ################
        # buttons
        # ################

        convert_bn = Qt.QPushButton('Convert')
        cancel_bn = Qt.QPushButton('Cancel')
        h_layout = Qt.QHBoxLayout()
        topLayout.addLayout(h_layout, 4, 0, 1, 2,
                            Qt.Qt.AlignHCenter | Qt.Qt.AlignTop)
        h_layout.addWidget(convert_bn)
        h_layout.addWidget(cancel_bn)

        # #################
        # setting initial state
        # #################

        self.__converter = None

        # named tuple with references to all the important widgets
        SelfWidgets = namedtuple('SelfWidgets',
                                 ['h5_file_edit',
                                  'h5_file_bn',
                                  'scans_gbx',
                                  'conv_gbx',
                                  'output_gbx',
                                  'xMinText',
                                  'xMaxText',
                                  'yMinText',
                                  'yMaxText',
                                  'n_img_label',
                                  'n_angles_label',
                                  'scans_table',
                                  'conv_params_wid',
                                  'output_file_edit',
                                  'output_file_bn',
                                  'convert_bn'])
        self.__widgets = SelfWidgets(h5_file_edit=h5_file_edit,
                                     h5_file_bn=h5_file_bn,
                                     scans_gbx=scans_gbx,
                                     conv_gbx=conv_gbx,
                                     output_gbx=output_gbx,
                                     xMinText=xMinText,
                                     xMaxText=xMaxText,
                                     yMinText=yMinText,
                                     yMaxText=yMaxText,
                                     n_img_label=n_img_label,
                                     n_angles_label=n_angles_label,
                                     scans_table=scans_table,
                                     conv_params_wid=conv_params_wid,
                                     output_file_edit=output_file_edit,
                                     output_file_bn=output_file_bn,
                                     convert_bn=convert_bn)

        cancel_bn.clicked.connect(self.close)
        h5_file_bn.clicked.connect(self.__pickInputFile)
        output_file_bn.clicked.connect(self.__pickOutputFile)
        convert_bn.clicked.connect(self.__convertBnClicked)

        self.__resetState()

        if qspace_size is not None:
            conv_params_wid.qspace_size = qspace_size

        if image_binning is not None:
            conv_params_wid.image_binning = image_binning

        if data_h5f is not None:
            h5_file_edit.setText(data_h5f)
            self.__readInputFile()

        if output_f is not None:
            output_file_edit.setText(output_f)

    def __convertBnClicked(self, checked):
        widgets = self.__widgets
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

        output_file = widgets.output_file_edit.text()

        if len(output_file) == 0:
            Qt.QMessageBox.critical(self, 'Error',
                                    'Output file field is mandatory.')
            return

        image_binning = widgets.conv_params_wid.image_binning
        qspace_size = widgets.conv_params_wid.qspace_size

        try:
            converter.image_binning = image_binning
            converter.qspace_size = qspace_size
        except ValueError as ex:
            Qt.QMessageBox.critical(self, 'Error',
                                    str(ex))
            return

        converter.output_f = output_file
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
        procDialog.accepted.connect(partial(
            self.__convertDone, status=RecipSpaceWidget.StatusCompleted))
        procDialog.rejected.connect(partial(
            self.__convertDone, status=RecipSpaceWidget.StatusAborted))
        self._setStatus(self.StatusRunning)
        procDialog.exec_()

    def __convertDone(self, status=None):
        self._setStatus(status)
        if status == RecipSpaceWidget.StatusCompleted:
            self.__qspaceH5 = self.__widgets.output_file_edit.text()
            self.hide()
        else:
            self.__qspaceH5 = None
        processedData = self._processData()
        self.sigProcessDone.emit(processedData)

    qspaceH5 = property(lambda self: self.__qspaceH5)

    def _processData(self):
        return self.qspaceH5

    status = property(lambda self: self.__status)

    def _setStatus(self, status):
        if status not in RecipSpaceWidget.StatusList:
            raise ValueError('Unknown status value : {0}.'
                             ''.format(status))
        self.__status = status

    def __resetState(self):
        widgets = self.__widgets

        widgets.scans_table.clear()
        widgets.scans_table.setHorizontalHeaderLabels(['scan', 'eta'])

        widgets.n_img_label.setText('')
        widgets.n_angles_label.setText('')

        self.__groupsSetEnabled(False)

        widgets.output_file_edit.setText('')

        self._setStatus(RecipSpaceWidget.StatusInit)

    def __pickOutputFile(self, checked):
        """
        output HDF5 file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select output file',
                                filter=('HDF5 files (*.h5);;'
                                        'Any files (*)'))
        dialog.setFileMode(Qt.QFileDialog.AnyFile)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            widgets.output_file_edit.setText(file_name)

    def __pickInputFile(self, checked):
        """
        HDF5 file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select input file',
                                filter=('HDF5 files (*.h5);;'
                                        'Any files (*)'))
        dialog.setFileMode(Qt.QFileDialog.ExistingFile)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            widgets.h5_file_edit.setText(file_name)
            self.__readInputFile()

    def __readInputFile(self):
        """
        Reads the input file and updates the GUI
        """
        widgets = self.__widgets
        converter = self.__converter

        if converter is not None and converter.is_running():
            raise ValueError('TODO : there is a conversion running.')

        self.__resetState()

        self.__converter = None

        input_f = str(widgets.h5_file_edit.text())

        if len(input_f) == 0:
            self.__resetState()

        # TODO : catch exceptions and popup errors
        try:
            converter = RecipSpaceConverter(input_f)
        except Exception as ex:
            print('EX : {0}.'.format(ex))
            raise ex

        self.__converter = converter

        if self.__rectRoi is not None:
            converter.rect_roi = self.__rectRoi

        self.__fillScansInfos()
        self.__groupsSetEnabled(True)

    def __fillScansInfos(self):
        """
        Fills the QTableWidget with info found in the input file
        """
        converter = self.__converter
        if converter is None:
            return

        widgets = self.__widgets
        scans = converter.scans
        scans_table = widgets.scans_table
        scans_table.setRowCount(len(scans))
        for row, scan in enumerate(scans):
            params = converter.scan_params(scan)
            item = Qt.QTableWidgetItem(scan)
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            scans_table.setItem(row, 0, item)
            item = Qt.QTableWidgetItem(str(params['angle']))
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            scans_table.setItem(row, 1, item)

        scans_table.resizeColumnsToContents()
        width = (sum([scans_table.columnWidth(i)
                     for i in range(scans_table.columnCount())]) +
                 scans_table.verticalHeader().width() +
                 20)
        # TODO : the size is wrong when the
        # verticalScrollBar isnt displayed yet
        # scans_table.verticalScrollBar().width())
        size = scans_table.minimumSize()
        size.setWidth(width)
        scans_table.setMinimumSize(size)

        # TODO : warning if the ROI is empty (too small to contain images)
        params = converter.scan_params(scans[0])
        rect_roi = converter.rect_roi
        if rect_roi is None:
            xMin = xMax = yMin = yMax = 'ns'
        else:
            xMin, xMax, yMin, yMax = rect_roi

        widgets.xMinText.setText(str(xMin))
        widgets.xMaxText.setText(str(xMax))
        widgets.yMinText.setText(str(yMin))
        widgets.yMaxText.setText(str(yMax))
        
        indices = converter.pos_indices
        nImgTxt = '{0} / {1}'.format(len(indices),
                                     params['n_images'])
        widgets.n_img_label.setText(nImgTxt)
        widgets.n_angles_label.setText(str(len(scans)))

    def __groupsSetEnabled(self, enable=True):
        widgets = self.__widgets
        widgets.scans_gbx.setEnabled(enable)
        widgets.conv_gbx.setEnabled(enable)
        widgets.output_gbx.setEnabled(enable)
        widgets.convert_bn.setEnabled(enable)


class _ConversionProcessDialog(Qt.QDialog):
    __sigConvertDone = Qt.Signal()

    def __init__(self, converter,
                 parent=None,
                 **kwargs):
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
        self.__status_lab.setText('<font color="orange">Cancelling...</font>')
        self.__bn_box.button(Qt.QDialogButtonBox.Abort).setEnabled(False)
        self.__converter.abort(wait=False)
        self.__aborted = True

    def __onProgress(self):
        progress = self.__converter.progress()
        self.__progress_bar.setValue(progress)

    def __convertDone(self):
        self.__qtimer.stop()
        self.__qtimer = None
        self.__onProgress()
        abortBn = self.__bn_box.button(Qt.QDialogButtonBox.Abort)
        if self.__aborted:
            self.__bn_box.rejected.disconnect(self.__onAbort)
            self.__status_lab.setText('<font color="red">Conversion '
                                      'cancelled.</font>')
            abortBn.setText('Close')
            self.__bn_box.rejected.connect(self.reject)
            abortBn.setEnabled(True)
        else:
            self.__bn_box.removeButton(abortBn)
            okBn = self.__bn_box.addButton(Qt.QDialogButtonBox.Ok)
            self.__status_lab.setText('<font color="green">Conversion '
                                      'done.</font>')
            okBn.setText('Close')

    status = property(lambda self: 0 if self.__aborted else 1)


if __name__ == '__main__':
    pass
