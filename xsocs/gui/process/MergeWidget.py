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
import time
import shutil
from types import MethodType
from functools import partial

from ...process.merge import KmapSpecParser, KmapMerger

from ..widgets.AcqParamsWidget import AcqParamsWidget

from ..widgets.Input import StyledLineEdit
from ..widgets.Containers import GroupBox
from ..widgets.FileChooser import FileChooser
from ..widgets.Buttons import FixedSizePushButon

from silx.gui import qt as Qt

_HELP_WIDGET_STYLE = """
            QLabel {
                border-radius: 10px;
                padding: 1px 4px;
                background-color: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 rgba(0, 0, 255, 255), stop:1 rgba(255, 255, 255, 255));
                color: rgb(255, 255, 255);
            }"""  # noqa


def _vLine(*args, **kwargs):
    vLine = Qt.QFrame(*args, **kwargs)
    vLine.setFrameShape(Qt.QFrame.VLine)
    vLine.setFrameShadow(Qt.QFrame.Sunken)
    return vLine


def _create_tmp_dir():

    qt_tmp_tpl = os.path.join(Qt.QDir.tempPath(),
                              'tmpXsocsXXXXXX')
    tmp_dir = delete_tmp = q_tmp_dir = None
    try:
        q_tmp_dir = Qt.QTemporaryDir(qt_tmp_tpl)
        isValid = q_tmp_dir.isValid()
        delete_tmp = False
        tmp_dir = q_tmp_dir.path()
        q_tmp_dir.setAutoRemove(False)
    except AttributeError:
        isValid = False

    if not isValid:
        q_tmp_dir = None
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        delete_tmp = True

    return tmp_dir, delete_tmp, q_tmp_dir


class _ScansSelectDialog(Qt.QDialog):
    
    (SEL_COL, ID_COL,
     M0_COL, M0_START_COL, M0_END_COL, M0_STEP_COL,
     M1_COL, M1_START_COL, M1_END_COL, M1_STEP_COL,
     IMG_FILE_COL, COL_COUNT) = range(12)

    def __init__(self, merger, **kwargs):
        super(_ScansSelectDialog, self).__init__(**kwargs)
        layout = Qt.QGridLayout(self)

        matched = merger.matched_ids
        selected = merger.selected_ids

        table_widget = Qt.QTableWidget(len(matched), self.COL_COUNT)
        table_widget.setHorizontalHeaderLabels(['', 'ID',
                                                'M0', 'start', 'end', 'step',
                                                'M1', 'start', 'end', 'step',
                                                'Image File'])

        def _sizeHint(self):
            width = (sum([self.columnWidth(i)
                     for i in range(self.columnCount())]) +
                     self.verticalHeader().width() +
                     20)
            return Qt.QSize(width, self.height())
        table_widget.sizeHint = MethodType(_sizeHint, table_widget)
        table_widget.minimumSize = MethodType(_sizeHint, table_widget)
        table_widget.maximumSize = MethodType(_sizeHint, table_widget)
        self.setSizePolicy(Qt.QSizePolicy(Qt.QSizePolicy.Fixed,
                                          Qt.QSizePolicy.Minimum))

        for num, scan_id in enumerate(matched):
            command = merger.get_scan_command(scan_id)

            item = Qt.QTableWidgetItem()
            item.setFlags(Qt.Qt.ItemIsUserCheckable |
                          Qt.Qt.ItemIsEditable |
                          Qt.Qt.ItemIsSelectable |
                          Qt.Qt.ItemIsEnabled)
            state = Qt.Qt.Checked if scan_id in selected else Qt.Qt.Unchecked
            item.setCheckState(state)
            table_widget.setItem(num, self.SEL_COL, item)

            def _add_col(value, col_idx):
                item = Qt.QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
                item.setTextAlignment(Qt.Qt.AlignRight)
                table_widget.setItem(num, col_idx, item)

            _add_col(str(scan_id), self.ID_COL)
            _add_col(command['motor_0'], self.M0_COL)
            _add_col(command['motor_0_start'], self.M0_START_COL)
            _add_col(command['motor_0_end'], self.M0_END_COL)
            _add_col(command['motor_0_steps'], self.M0_STEP_COL)
            _add_col(command['motor_1'], self.M1_COL)
            _add_col(command['motor_1_start'], self.M1_START_COL)
            _add_col(command['motor_1_end'], self.M1_END_COL)
            _add_col(command['motor_1_steps'], self.M1_STEP_COL)

            img_file = merger.get_scan_image(scan_id)
            item = Qt.QTableWidgetItem(os.path.basename(img_file))
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            item.setToolTip(img_file)
            table_widget.setItem(num, self.IMG_FILE_COL, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        layout.addWidget(table_widget, 0, 0, Qt.Qt.AlignLeft)

        table_widget.setColumnHidden(self.M0_COL, True)
        table_widget.setColumnHidden(self.M0_START_COL, True)
        table_widget.setColumnHidden(self.M0_END_COL, True)
        table_widget.setColumnHidden(self.M0_STEP_COL, True)
        table_widget.setColumnHidden(self.M1_COL, True)
        table_widget.setColumnHidden(self.M1_START_COL, True)
        table_widget.setColumnHidden(self.M1_END_COL, True)
        table_widget.setColumnHidden(self.M1_STEP_COL, True)

        bnLayout = Qt.QGridLayout()
        layout.addLayout(bnLayout, 1, 0)

        selBn = Qt.QPushButton('Select')
        unselBn = Qt.QPushButton('Unselect')
        bnLayout.addWidget(selBn, 0, 0, Qt.Qt.AlignLeft)
        bnLayout.addWidget(unselBn, 0, 1, Qt.Qt.AlignLeft)
        selBn.clicked.connect(self.__selectClicked)
        unselBn.clicked.connect(self.__unselectClicked)
        bnLayout.setColumnStretch(2, 1)

        more_bn = FixedSizePushButon('More')
        bnLayout.addWidget(more_bn, 0, 3, Qt.Qt.AlignRight)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Apply')

        layout.addWidget(bn_box, 2, 0)
        bn_box.accepted.connect(self.__onAccept)
        bn_box.rejected.connect(self.reject)
        more_bn.clicked.connect(self.__showMore)

        self.__table_widget = table_widget
        self.__more_bn = more_bn
        self.__merger = merger

    def __selectClicked(self):
        indices = self.__table_widget.selectionModel().selectedIndexes()
        if len(indices) > 0:
            rows = set()
            for index in indices:
                rows.add(index.row())
            for row in rows:
                item = self.__table_widget.item(row, self.SEL_COL)
                item.setCheckState(Qt.Qt.Checked)

    def __unselectClicked(self):
        indices = self.__table_widget.selectionModel().selectedIndexes()
        if len(indices) > 0:
            rows = set()
            for index in indices:
                rows.add(index.row())
            for row in rows:
                item = self.__table_widget.item(row, self.SEL_COL)
                item.setCheckState(Qt.Qt.Unchecked)

    def __showMore(self):
        if self.__more_bn.text() == 'More':
            self.__more_bn.setText('Less')
            hide = False
        else:
            self.__more_bn.setText('More')
            hide = True
        table_widget = self.__table_widget
        table_widget.setColumnHidden(self.M0_COL, hide)
        table_widget.setColumnHidden(self.M0_START_COL, hide)
        table_widget.setColumnHidden(self.M0_END_COL, hide)
        table_widget.setColumnHidden(self.M0_STEP_COL, hide)
        table_widget.setColumnHidden(self.M1_COL, hide)
        table_widget.setColumnHidden(self.M1_START_COL, hide)
        table_widget.setColumnHidden(self.M1_END_COL, hide)
        table_widget.setColumnHidden(self.M1_STEP_COL, hide)
        table_widget.resizeColumnsToContents()
        table_widget.updateGeometry()
        self.adjustSize()

    def __onAccept(self, *args, **kwags):
        table_widget = self.__table_widget
        rowCount = table_widget.rowCount()
        selected = []
        for row in range(rowCount):
            sel_item = table_widget.item(row, 0)
            if sel_item.checkState() == Qt.Qt.Checked:
                id_item = table_widget.item(row, 1)
                selected.append(id_item.text())
        self.__merger.select(selected, clear=True)
        self.accept()


class _ScansInfoDialog(Qt.QDialog):

    def __init__(self, merger, **kwargs):
        super(_ScansInfoDialog, self).__init__(**kwargs)
        layout = Qt.QVBoxLayout(self)

        no_match = merger.no_match_ids
        no_img = merger.no_img_ids

        table_widget = Qt.QTableWidget(len(no_match) + len(no_img), 2)

        for num, scan_id in enumerate(no_match):
            item = Qt.QTableWidgetItem(scan_id)
            table_widget.setItem(num, 0, item)

            item = Qt.QTableWidgetItem('Image file not found.')
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            table_widget.setItem(num, 1, item)

        offset = len(no_match)

        for num, scan_id in enumerate(no_img):
            item = Qt.QTableWidgetItem(scan_id)
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            table_widget.setItem(num + offset, 0, item)

            item = Qt.QTableWidgetItem('No image info in header.')
            item.setFlags(item.flags() ^ Qt.Qt.ItemIsEditable)
            table_widget.setItem(num + offset, 1, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        table_widget.sortByColumn(0, Qt.Qt.AscendingOrder)

        layout.addWidget(table_widget)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Close)

        layout.addWidget(bn_box)
        bn_box.rejected.connect(self.reject)


class _MergeProcessDialog(Qt.QDialog):
    __sigMergeDone = Qt.Signal()

    def __init__(self, merger, **kwargs):
        super(_MergeProcessDialog, self).__init__(**kwargs)
        layout = Qt.QVBoxLayout(self)

        files = merger.summary()
        output_dir = merger.output_dir

        label = Qt.QLabel('<html><head/><body><p align="center">'
                          '<span style=" font-size:16pt; font-weight:600;">'
                          'Merge process</span></p></body></html>')
        label.setTextFormat(Qt.Qt.RichText)
        layout.addWidget(label, stretch=0, alignment=Qt.Qt.AlignHCenter)

        grp_box = GroupBox('Output directory :')
        grp_box.setLayout(Qt.QVBoxLayout())
        outdir_edit = Qt.QLineEdit(output_dir)
        fm = outdir_edit.fontMetrics()
        outdir_edit.setMinimumWidth(fm.width(' ' * 100))
        grp_box.layout().addWidget(outdir_edit)

        layout.addWidget(grp_box, stretch=0)
        grp_box = GroupBox('Files :')
        grp_box.setLayout(Qt.QVBoxLayout())
        tree_widget = Qt.QTreeWidget()
        tree_widget.setColumnCount(3)
        tree_widget.setColumnHidden(2, True)
        # TODO improve
        master_item = Qt.QTreeWidgetItem([files['master'], '', 'master'])
        for scan_id in sorted(files.keys()):
            if scan_id != 'master':
                master_item.addChild(Qt.QTreeWidgetItem([files[scan_id],
                                                         '',
                                                         scan_id]))

        tree_widget.addTopLevelItem(master_item)
        tree_widget.setItemWidget(master_item, 1, Qt.QProgressBar())
        for i_child in range(master_item.childCount()):
            tree_widget.setItemWidget(master_item.child(i_child),
                                      1,
                                      Qt.QProgressBar())

        master_item.setExpanded(True)
        tree_widget.resizeColumnToContents(0)
        tree_widget.resizeColumnToContents(1)
        width = (tree_widget.sizeHintForColumn(0) +
                 tree_widget.sizeHintForColumn(1) + 10)
        tree_widget.setMinimumWidth(width)
        layout.addWidget(tree_widget, stretch=1, alignment=Qt.Qt.AlignHCenter)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Merge')
        layout.addWidget(bn_box)
        bn_box.accepted.connect(self.__onAccept)
        bn_box.rejected.connect(self.reject)
        self.__sigMergeDone.connect(self.__mergeDone)

        self.__tree_widget = tree_widget
        self.__bn_box = bn_box
        self.__abort_diag = None
        self.__merger = merger
        self.__status = False

    def __onAccept(self):
        merger = self.__merger
        warn = merger.check_overwrite()

        if warn:
            ans = Qt.QMessageBox.warning(self,
                                         'Overwrite?',
                                         ('Some files already exist.'
                                          '\nDo you want to overwrite them?'),
                                         buttons=Qt.QMessageBox.Yes |
                                         Qt.QMessageBox.No)
            if ans == Qt.QMessageBox.No:
                return

        self.__mergeStart()

    def __mergeStart(self):
        self.__bn_box.rejected.disconnect(self.reject)
        self.__bn_box.rejected.connect(self.__onAbort)
        self.__bn_box.button(Qt.QDialogButtonBox.Ok).setEnabled(False)
        self.__bn_box.button(Qt.QDialogButtonBox.Cancel).setText('Abort')

        self.__qtimer = Qt.QTimer()
        self.__qtimer.timeout.connect(self.__onProgress)
        self.__merger.merge(overwrite=True,
                            blocking=False,
                            callback=self.__sigMergeDone.emit)
        self.__onProgress()
        self.__qtimer.start(1000)

        self.__time = time.time()

    def __onAbort(self):
        abort_diag = Qt.QMessageBox(Qt.QMessageBox.Information,
                                    'Aborting...',
                                    '<b>Cancelling merge.</b>'
                                    '<center>Please wait...</center>',
                                    parent=self)
        self.__abort_diag = abort_diag
        abort_diag.setTextFormat(Qt.Qt.RichText)
        abort_diag.setAttribute(Qt.Qt.WA_DeleteOnClose)
        abort_diag.setStandardButtons(Qt.QMessageBox.NoButton)
        abort_diag.show()
        self.__merger.abort(wait=False)

    def __mergeDone(self):
        print('TOTAL : {0}.'.format(time.time() - self.__time))
        self.__status = self.__merger.status
        self.__qtimer.stop()
        self.__qtimer = None
        self.__onProgress()
        if self.__abort_diag is not None:
            self.__abort_diag.done(0)
            self.__abort_diag = None
        self.__bn_box.button(Qt.QDialogButtonBox.Cancel).setText('Close')

        if self.__status:
            self.__bn_box.rejected.connect(self.accept)
        else:
            self.__bn_box.rejected.connect(self.reject)

    def __onProgress(self):
        progress = self.__merger.progress()
        if progress is None:
            return
        tree_wid = self.__tree_widget
        flags = Qt.Qt.MatchExactly | Qt.Qt.MatchRecursive
        total = 0.
        for scan_id, prog in progress.items():
            total += prog
            item = tree_wid.findItems(scan_id,
                                      flags,
                                      column=2)
            if len(item) > 0:
                item = item[0]
                wid = tree_wid.itemWidget(item, 1)
                wid.setValue(prog)
        item = tree_wid.findItems('master',
                                  flags,
                                  column=2)
        if len(item) > 0:
            item = item[0]
            wid = tree_wid.itemWidget(item, 1)
            wid.setValue(total / len(progress))


class MergeWidget(Qt.QDialog):

    _versions = {'padding': [4, 5],
                 'offset': [-1, 0]}
    _defaultVersion = 1

    __sigParsed = Qt.Signal()

    def __init__(self,
                 spec_file=None,
                 img_dir=None,
                 spec_version=1,
                 output_dir=None,
                 tmp_dir=None,
                 **kwargs):
        super(MergeWidget, self).__init__(**kwargs)

        Qt.QGridLayout(self)

        # ################
        # input QGroupBox
        # ################

        if spec_file is not None:
            specFile = spec_file
        else:
            specFile = ''

        if img_dir is not None:
            imgDir = img_dir
        else:
            imgDir = ''

        if spec_version is not None:
            specVersion = spec_version
        else:
            specVersion = self._defaultVersion

        if output_dir is not None:
            outputDir = output_dir
        else:
            outputDir = ''

        # parameters
        self.__input = {'specfile': specFile,
                        'imgdir': imgDir,
                        'version': specVersion,
                        'padding': None,
                        'offset': None}

        self.__output = {'outdir': outputDir,
                         'prefix': ''}

        inputGbx = GroupBox("Input")
        layout = Qt.QGridLayout(inputGbx)
        self.layout().addWidget(inputGbx,
                                0, 0,
                                Qt.Qt.AlignTop)

        first_col = 0
        file_bn_col = 4
        last_col = file_bn_col + 1

        spec_row = 0
        img_path_row = 1
        version_row = 2
        apply_bn_row = 3

        # spec file input
        specFileChooser = FileChooser(fileMode=Qt.QFileDialog.ExistingFile,
                                      noLabel=True)
        specFileChooser.lineEdit.setText(specFile)
        specFileChooser.sigSelectionChanged.connect(self.__slotSpecFileChanged)
        layout.addWidget(Qt.QLabel('Spec file :'), spec_row, 0)
        layout.addWidget(specFileChooser, spec_row, 1)

        # image folder input
        imgDirChooser = FileChooser(fileMode=Qt.QFileDialog.Directory,
                                    noLabel=True)
        imgDirChooser.lineEdit.setText(imgDir)
        imgDirChooser.sigSelectionChanged.connect(self.__slotImgDirChanged)
        layout.addWidget(Qt.QLabel('Img dir. :'), img_path_row, 0)
        layout.addWidget(imgDirChooser, img_path_row, 1)

        # version selection
        optionLayout = Qt.QHBoxLayout()
        optionLayout.addStretch(1)

        lab = Qt.QLabel('Version :')
        self.__versionCBx = versionCBx = Qt.QComboBox()

        for version in range(len(self._versions)):
            versionCBx.addItem(str(version))
        versionCBx.addItem('')
        optionLayout.addWidget(lab, Qt.Qt.AlignLeft)
        optionLayout.addWidget(versionCBx, Qt.Qt.AlignLeft)

        # filename padding for the nextNr counter
        self.__padSpinBox = padSpinBox = Qt.QSpinBox()
        optionLayout.addWidget(_vLine())
        optionLayout.addWidget(Qt.QLabel('nextNr padding:'))
        optionLayout.addWidget(padSpinBox, Qt.Qt.AlignLeft)
        padSpinBox.valueChanged[int].connect(self.__slotPaddingValueChanged)

        # filename offset for the nextNr counter
        self.__nextNrSpinBox = nextNrSpinBox = Qt.QSpinBox()
        nextNrSpinBox.setMinimum(-100)
        nextNrSpinBox.setMaximum(100)
        optionLayout.addWidget(_vLine())
        optionLayout.addWidget(Qt.QLabel('nextNr offset:'))
        optionLayout.addWidget(nextNrSpinBox, Qt.Qt.AlignLeft)
        nextNrSpinBox.valueChanged[int].connect(self.__slotNextNrValueChanged)

        optionLayout.addStretch(100)
        layout.addLayout(optionLayout, version_row, 0,
                         1, layout.columnCount(),
                         Qt.Qt.AlignLeft)

        # last row : apply button
        self.__parseBn = parseBn = FixedSizePushButon('Parse file')
        parseBn.clicked.connect(self.__slotParseBnClicked,
                                Qt.Qt.QueuedConnection)
        layout.addWidget(parseBn,
                         apply_bn_row, 0,
                         1, last_col - first_col,
                         Qt.Qt.AlignHCenter)

        # ################
        # scans + edf QGroupBox
        # ################
        self.__scansGbx = scansGbx = GroupBox("Spec + EDF")
        grpLayout = Qt.QHBoxLayout(scansGbx)
        self.layout().addWidget(scansGbx,
                                1, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        # ===========
        # valid scans
        # ===========
        scanLayout = Qt.QGridLayout()
        grpLayout.addLayout(scanLayout)

        hLayout = Qt.QHBoxLayout()
        label = Qt.QLabel('<span style=" font-weight:600; color:#00916a;">'
                          'Matched scans</span>')
        label.setTextFormat(Qt.Qt.RichText)
        editScansBn = FixedSizePushButon('Edit')
        editScansBn.clicked.connect(self.__slotEditScansClicked)
        hLayout.addWidget(label)
        hLayout.addWidget(editScansBn)
        scanLayout.addLayout(hLayout, 0, 0, 1, 2)

        label = Qt.QLabel('Total :')
        self.__totalScansEdit = totalScansEdit = Qt.QLineEdit('0')
        totalScansEdit.setReadOnly(True)
        fm = totalScansEdit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        totalScansEdit.setMaximumWidth(width)
        totalScansEdit.setAlignment(Qt.Qt.AlignRight)

        scanLayout.addWidget(label, 1, 0, Qt.Qt.AlignLeft)
        scanLayout.addWidget(totalScansEdit, 1, 1, Qt.Qt.AlignLeft)

        # ====

        label = Qt.QLabel('Selected :')
        self.__selectedScansEdit = selectedScansEdit = Qt.QLineEdit('0')
        selectedScansEdit.setReadOnly(True)
        fm = selectedScansEdit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        selectedScansEdit.setMaximumWidth(width)
        selectedScansEdit.setAlignment(Qt.Qt.AlignRight)

        scanLayout.addWidget(label, 2, 0, Qt.Qt.AlignLeft)
        scanLayout.addWidget(selectedScansEdit, 2, 1, Qt.Qt.AlignLeft)

        # ===

        grpLayout.addWidget(_vLine())

        # ===========
        # "other" scans
        # ===========

        scanLayout = Qt.QGridLayout()
        grpLayout.addLayout(scanLayout)

        hLayout = Qt.QHBoxLayout()
        label = Qt.QLabel('<span style=" font-weight:600; color:#ff6600;">'
                          'Other scans</span>')
        otherScansBn = FixedSizePushButon('View')
        otherScansBn.clicked.connect(self.__slotOtherScansClicked)
        hLayout.addWidget(label)
        hLayout.addWidget(otherScansBn)

        scanLayout.addLayout(hLayout, 0, 0, 1, 2)

        label = Qt.QLabel('No match :')
        self.__noMatchScansEdit = noMatchScansEdit = Qt.QLineEdit('0')
        noMatchScansEdit.setReadOnly(True)
        fm = noMatchScansEdit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        noMatchScansEdit.setMaximumWidth(width)
        noMatchScansEdit.setAlignment(Qt.Qt.AlignRight)

        scanLayout.addWidget(label, 1, 0, Qt.Qt.AlignLeft)
        scanLayout.addWidget(noMatchScansEdit, 1, 1, Qt.Qt.AlignLeft)

        # ====

        label = Qt.QLabel('No img info :')
        self.__noImgInfoEdit = noImgInfoEdit = Qt.QLineEdit('0')
        noImgInfoEdit.setReadOnly(True)
        fm = noImgInfoEdit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        noImgInfoEdit.setMaximumWidth(width)
        noImgInfoEdit.setAlignment(Qt.Qt.AlignRight)

        scanLayout.addWidget(label, 2, 0, Qt.Qt.AlignLeft)
        scanLayout.addWidget(noImgInfoEdit, 2, 1, Qt.Qt.AlignLeft)

        # ===

        grpLayout.addWidget(_vLine())

        # ################
        # parameters
        # ################
        self.__acqParamsGbx = acqParamsGbx = GroupBox("Acq. Parameters")
        grpLayout = Qt.QVBoxLayout(acqParamsGbx)

        self.__acqParamWid = acqParamWid = AcqParamsWidget()
        self.layout().addWidget(acqParamsGbx,
                                2, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        grpLayout.addWidget(acqParamWid)

        # ################
        # output options
        # ################

        self.__outputGbx = outputGbx = GroupBox("Output")
        layout = Qt.QGridLayout(outputGbx)
        self.layout().addWidget(outputGbx,
                                3, 0,
                                Qt.Qt.AlignTop)

        # ===========
        # master
        # ===========

        lab = Qt.QLabel('Prefix :')
        self.__prefixEdit = prefixEdit = StyledLineEdit(nChar=20)
        prefixEdit.textChanged.connect(self.__slotPrefixChanged)
        hLayout = Qt.QHBoxLayout()
        layout.addLayout(hLayout, 0, 1, Qt.Qt.AlignLeft)
        resetPrefixBn = Qt.QToolButton()
        icon = Qt.qApp.style().standardIcon(Qt.QStyle.SP_BrowserReload)
        resetPrefixBn.setIcon(icon)
        resetPrefixBn.clicked.connect(self.__slotResetPrefixClicked)
        layout.addWidget(lab,
                         0, 0,
                         Qt.Qt.AlignLeft)
        sp = prefixEdit.sizePolicy()
        sp.setHorizontalPolicy(Qt.QSizePolicy.Maximum)
        prefixEdit.setSizePolicy(sp)
        hLayout.addWidget(prefixEdit, Qt.Qt.AlignLeft)
        hLayout.addWidget(resetPrefixBn, Qt.Qt.AlignLeft)

        # ===========
        # output folder
        # ===========

        outDirChooser = FileChooser(fileMode=Qt.QFileDialog.Directory,
                                    noLabel=True)
        outDirChooser.lineEdit.setText(outputDir)
        imgDirChooser.sigSelectionChanged.connect(self.__slotOutDirChanged)
        layout.addWidget(Qt.QLabel('Output directory :'), 1, 0)
        layout.addWidget(outDirChooser, 1, 1)

        # ################
        # merge button
        # ################

        self.__mergeBn = mergeBn = Qt.QPushButton('Merge')
        cancelBn = Qt.QPushButton('Cancel')
        hLayout = Qt.QHBoxLayout()
        self.layout().addLayout(hLayout,
                                4, 0,
                                1, 1,
                                Qt.Qt.AlignHCenter | Qt.Qt.AlignTop)
        hLayout.addWidget(mergeBn)
        hLayout.addWidget(cancelBn)
        mergeBn.clicked.connect(self.__slotMergeBnClicked)
        cancelBn.clicked.connect(self.reject)

        # #################
        # setting initial state
        # #################

        # scansGbx.setEnabled(False)
        # acqParamsGbx.setEnabled(False)
        # outputGbx.setEnabled(False)
        # mergeBn.setEnabled(False)
        parseBn.setEnabled(False)

        self.__merger = None
        self.__parser = None
        self.info_wid = None

        if tmp_dir is None:
            tmp_dir, delete_tmp, q_tmp_dir = _create_tmp_dir()
        else:
            delete_tmp = False
            q_tmp_dir = None

        self.__tmp_root = tmp_dir
        self.__delete_tmp_root = delete_tmp
        self.__q_tmp_dir = q_tmp_dir

        tmp_dir = os.path.join(self.__tmp_root, 'xsocs_merge')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.__tmp_dir_merge = tmp_dir

        print('Using temporary folder : {0}.'.format(tmp_dir))

        self.__widgetIsSetup = False

        self.__xsocs_h5 = None

        versionCBx.setCurrentIndex(spec_version)
        self.__slotVersionChanged(spec_version)
        versionCBx.currentIndexChanged[int].connect(self.__slotVersionChanged)

        self.__resetState()

        self.__sigParsed.connect(self.__slotParsed, Qt.Qt.QueuedConnection)

    def showEvent(self, *args, **kwargs):
        if not self.__widgetIsSetup:
            self.__widgetIsSetup = True
            specFile = self.__input['specfile']
            self.__checkInput()
            if specFile:
                self.__parseBn.clicked.emit(True)
        super(MergeWidget, self).showEvent(*args, **kwargs)

    def closeEvent(self, event):
        if self.__delete_tmp_root and os.path.exists(self.__tmp_root):
            shutil.rmtree(self.__tmp_root, ignore_errors=True)
        elif os.path.exists(self.__tmp_dir_merge):
            shutil.rmtree(self.__tmp_dir_merge, ignore_errors=True)
        if self.__q_tmp_dir is not None:
            # for some reason the QTemporaryDir gets deleted even though
            # this instance is still in scope. This is a workaround until
            # we figure out what's going on.
            # (deletion seems to occur when creating the Pool in the
            # _MergeThread::run method)
            self.__q_tmp_dir.setAutoRemove(True)
        super(MergeWidget, self).closeEvent(event)

    def __slotVersionChanged(self, index):
        if index < 0 or index >= len(self._versions):
            self.__input['version'] = index
            self.__versionCBx.setCurrentIndex(len(self._versions))
        else:
            padding = self._versions['padding'][index]
            offset = self._versions['offset'][index]

            self.__input['version'] = index

            padSpinBox = self.__padSpinBox
            nextNrSpinBox = self.__nextNrSpinBox
            blocked = padSpinBox.blockSignals(True)
            padSpinBox.setValue(padding)
            self.__input['padding'] = padding
            padSpinBox.blockSignals(blocked)

            blocked = nextNrSpinBox.blockSignals(True)
            nextNrSpinBox.setValue(offset)
            self.__input['offset'] = offset
            nextNrSpinBox.blockSignals(blocked)

        self.__checkInput()

    def __slotResetPrefixClicked(self):
        """
        Slot called when the reset prefix button is clicked.
        :return:
        """
        merger = self.__merger
        if merger is None:
            return
        merger.prefix = None
        master = merger.prefix
        self.__prefixEdit.setText(master)

    def __slotMergeBnClicked(self):
        """
        Slot called when the merge button is clicked
        :return:
        """
        merger = self.__merger

        if merger is None:
            # this part shouldn't even be called, just putting this
            # in case someone decides to modify the code to enable the mergeBn
            # even tho conditions are not met.
            Qt.QMessageBox.critical(self, 'Error',
                                    'No merger object found.',
                                    'Has a SPEC file been parsed yet?')
            return

        if len(merger.selected_ids) == 0:
            Qt.QMessageBox.warning(self,
                                   'Selection error',
                                   'At least one scan has to be selected.')
            return

        def assert_non_none(val):
            if val is None:
                raise ValueError('parameter is mandatory.')
            return val

        acqParamWid = self.__acqParamWid

        name = None

        try:
            name = 'Beam Energy'
            merger.beam_energy = \
                assert_non_none(acqParamWid.beam_energy)

            name = 'Direct beam'
            dir_beam_h = assert_non_none(acqParamWid.direct_beam_h)
            dir_beam_v = assert_non_none(acqParamWid.direct_beam_v)
            merger.center_chan = [dir_beam_h, dir_beam_v]

            name = 'Channel per degree'
            chpdeg_h = assert_non_none(acqParamWid.chperdeg_h)
            chpdeg_v = assert_non_none(acqParamWid.chperdeg_v)
            merger.chan_per_deg = [chpdeg_h, chpdeg_v]

            name = 'Prefix'
            prefix = self.__output['prefix']
            if not prefix:
                raise ValueError('parameter is mandatory.')
            merger.prefix = str(prefix)

            name = 'Output directory'
            outDir = self.__output['outdir']
            if not outDir:
                raise ValueError('parameter is mandatory.')
            merger.output_dir = str(outDir)

        except Exception as ex:
            Qt.QMessageBox.critical(self, 'Error',
                                    '{0} : {1}.'.format(name, str(ex)))
            return

        param_errors = merger.check_parameters()
        if len(param_errors) > 0:
            txt = ('Please fix the following error(s) before merging :\n- {0}'
                   ''.format('- '.join(param_errors)))
            Qt.QMessageBox.critical(self, 'Parameters errors.', txt)
            return
        scans_errors = merger.check_consistency()
        if len(scans_errors) > 0:
            txt = ('Please fix the following error(s) before merging :\n- {0}'
                   ''.format('- '.join(scans_errors)))
            Qt.QMessageBox.critical(self, 'Selected scans errors.', txt)
            return

        process_diag = _MergeProcessDialog(merger, parent=self)
        process_diag.setAttribute(Qt.Qt.WA_DeleteOnClose)
        process_diag.accepted.connect(partial(self.__mergeDone))
        process_diag.rejected.connect(partial(self.__mergeDone))
        process_diag.setModal(True)
        self.__process_diag = process_diag
        process_diag.show()

    def __mergeDone(self):
        self.__process_diag = None
        status = self.__merger.status
        if status == KmapMerger.DONE:
            self.__xsocs_h5 = self.__merger.master_file
            self.accept()

    @property
    def xsocsH5(self):
        return self.__xsocs_h5

    def __slotSpecFileChanged(self, fileName):
        """
        Slot triggered when the spec file is changed.
        """
        self.__input['specfile'] = str(fileName)
        self.__checkInput()

    def __slotImgDirChanged(self, dirName):
        """
        Slot triggered when the image folder is changed.
        :param dirName:
        :return:
        """
        self.__input['imgdir'] = str(dirName)
        self.__checkInput()

    def __checkInput(self):
        """
        Checks if all input is provided and enables/disables the parse button
        and resets the parse results accordingly.
        :return:
        """
        specfile = self.__input['specfile']
        if specfile and os.path.isfile(specfile):
            enabled = True
        else:
            enabled = False

        imgdir = self.__input['imgdir']
        if imgdir and not os.path.isdir(imgdir):
            enabled = False

        self.__parseBn.setEnabled(enabled)

        self.__resetState()

    def __resetState(self):
        """
        Sets the default state for the groupboxes
        - disables all but the input groupbox
        - clears the scan widget
        """
        self.__scansGbx.setEnabled(False)
        self.__acqParamsGbx.setEnabled(False)
        self.__outputGbx.setEnabled(False)
        self.__mergeBn.setEnabled(False)
        self.__merger = None

    def __slotOutDirChanged(self, outDir):
        """
        Output dir file picker
        """
        self.__output['outdir'] = outDir
        self.__updateOutputGroupBox()

    def __slotEditScansClicked(self):
        """
        Slot called when the edit scans button is clicked
        :return:
        """
        merger = self.__merger
        if merger is None:
            return
        ans = _ScansSelectDialog(merger, parent=self).exec_()
        if ans == Qt.QDialog.Accepted:
            self.__updateScansInfos()

    def __slotOtherScansClicked(self):
        """
        Slot called when the other scans button is clicked
        :return:
        """
        merger = self.__merger
        if merger is None:
            return
        _ScansInfoDialog(merger, parent=self).exec_()

    def __slotPrefixChanged(self, text):
        """
        Slot called when the text in the prefixEdit line edit changes
        :param text:
        :return:
        """
        self.__output['prefix'] = str(text)
        self.__updateOutputGroupBox()

    def __updateOutputGroupBox(self):
        merger = self.__merger

        if merger is None:
            enable = False
        else:
            enable = len(merger.matched_ids) > 0

        self.__outputGbx.setEnabled(enable)
        if not enable:
            self.__prefixEdit.clear()

        outDir = self.__output['outdir']
        prefix = self.__output['prefix']
        hasOutputDir = outDir is not None and len(outDir) > 0
        hasPrefix = prefix is not None and len(prefix) > 0
        enable = hasOutputDir and hasPrefix
        self.__mergeBn.setEnabled(enable)

    def __slotPaddingValueChanged(self, value):
        """
        Slot called when the padding spinbox value changes
        :param value:
        :return:
        """
        self.__input['padding'] = value
        self.__slotVersionChanged(-1)

    def __slotNextNrValueChanged(self, value):
        """
        Slot called when the next nr offset spinbox value changes
        :param value:
        :return:
        """
        self.__input['offset'] = value
        self.__slotVersionChanged(-1)

    def __slotParseBnClicked(self):
        self.info_wid = None

        parseBn = self.__parseBn

        specFile = self.__input['specfile']
        imgDir = self.__input['imgdir']
        # version = self.__input['version']
        padding = self.__input['padding']
        offset = self.__input['offset']

        spec_h5 = os.path.join(self.__tmp_dir_merge, 'temp_spec.h5')

        try:
            self.__parser = KmapSpecParser(specFile,
                                           spec_h5,
                                           img_dir=imgDir,
                                           # version=version,
                                           nr_offset=offset,
                                           nr_padding=padding,
                                           callback=self.__sigParsed.emit)
        except Exception as ex:
            msg = ('Parsing failed: {0}.\n'
                   'Message : {1}.'
                   ''.format(ex.__class__.__name__, str(ex)))
            Qt.QMessageBox.critical(self,
                                    'Parse error.',
                                    msg)
            return

        parseBn.setEnabled(False)
        parseBn.setText('Parsing...')

        info_wid = Qt.QMessageBox(Qt.QMessageBox.Information,
                                  'Parsing...',
                                  '<b>Parsing SPEC file and matching image'
                                  ' files.</b>'
                                  '<center>Please wait...</center>',
                                  parent=self)
        info_wid.setTextFormat(Qt.Qt.RichText)
        info_wid.setAttribute(Qt.Qt.WA_DeleteOnClose)
        info_wid.setStandardButtons(Qt.QMessageBox.NoButton)
        info_wid.show()
        self.info_wid = info_wid
        self.__parser.parse(blocking=False)

    def __slotParsed(self):

        self.info_wid.done(0)
        self.info_wid = None

        self.__parseBn.setEnabled(True)
        self.__parseBn.setText('Parse file.')

        self.__merger = KmapMerger(self.__parser.results.spec_h5,
                                   self.__parser.results,
                                   output_dir=self.__output['outdir'])

        self.__output['prefix'] = self.__merger.prefix
        blocked = self.__prefixEdit.blockSignals(True)
        self.__prefixEdit.setText(self.__merger.prefix)
        self.__prefixEdit.blockSignals(blocked)

        self.__updateScansInfos()
        self.__updateOutputGroupBox()

    def __updateScansInfos(self):
        """
        Fills the scans group box with the results
        of the scan.
        """
        merger = self.__merger

        if merger is None:
            matched_ids = []
            selected_ids = []
            no_match_ids = []
            no_img_ids = []
            enable = False
            prefix = ''
        else:
            matched_ids = merger.matched_ids
            selected_ids = merger.selected_ids
            no_match_ids = merger.no_match_ids
            no_img_ids = merger.no_img_ids
            enable = True
            prefix = merger.prefix

        nTotal = len(matched_ids)
        nSelected = len(selected_ids)
        nNoMatch = len(no_match_ids)
        nNoImg = len(no_img_ids)

        self.__scansGbx.setEnabled(enable)
        self.__acqParamsGbx.setEnabled(len(matched_ids) > 0)

        self.__totalScansEdit.setText(str(nTotal))
        self.__selectedScansEdit.setText(str(nSelected))
        self.__noMatchScansEdit.setText(str(nNoMatch))
        self.__noImgInfoEdit.setText(str(nNoImg))

        self.__output['prefix'] = prefix
        self.__prefixEdit.setText(prefix)


if __name__ == '__main__':
    pass
