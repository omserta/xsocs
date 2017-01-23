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
from collections import namedtuple

from ...util.id01_spec import Id01DataMerger
from ..widgets.AcqParamsWidget import AcqParamsWidget

from ..widgets.Containers import GroupBox
from ..widgets.Buttons import FixedSizePushButon

from silx.gui import qt as Qt

_HELP_WIDGET_STYLE = """
            QLabel {
                border-radius: 10px;
                padding: 1px 4px;
                background-color: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 rgba(0, 0, 255, 255), stop:1 rgba(255, 255, 255, 255));
                color: rgb(255, 255, 255);
            }"""  # noqa


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

    def __showMore(self, *args, **kwargs):
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
    __sigMergeDone = Qt.Signal(object)

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

    def __onAccept(self, *args, **kwargs):
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

    def __onAbort(self, *args, **kwargs):
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
        self.__merger.abort_merge(wait=False)

    def __mergeDone(self, result):
        print('TOTAL : {0}.'.format(time.time() - self.__time))
        self.__status = result[0]
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

    def __onProgress(self, *args, **kwargs):
        progress = self.__merger.merge_progress()
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

    __sigParseDone = Qt.Signal()

    def __init__(self,
                 spec_file=None,
                 img_dir=None,
                 spec_version=None,
                 output_dir=None,
                 tmp_dir=None,
                 **kwargs):
        super(Qt.QWidget, self).__init__(**kwargs)
        Qt.QGridLayout(self)

        # ################
        # input QGroupBox
        # ################

        input_gbx = GroupBox("Input")
        layout = Qt.QGridLayout(input_gbx)
        self.layout().addWidget(input_gbx,
                                0, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        first_col = 0
        label_col = first_col
        line_edit_col = 1
        file_bn_col = 4
        vers_help_col = 2
        last_col = file_bn_col + 1

        spec_row = 0
        img_path_row = 1
        version_row = 2
        apply_bn_row = 3

        # spec file input
        lab = Qt.QLabel('Spec file :')
        spec_file_edit = Qt.QLineEdit()
        fm = spec_file_edit.fontMetrics()
        spec_file_edit.setMinimumWidth(fm.width(' ' * 100))
        spec_file_bn = FixedSizePushButon('...')
        layout.addWidget(lab,
                         spec_row, label_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(spec_file_edit,
                         spec_row, line_edit_col,
                         1, file_bn_col - line_edit_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(spec_file_bn,
                         spec_row, file_bn_col,
                         Qt.Qt.AlignLeft)

        # image folder input
        lab = Qt.QLabel('Img folder :')
        img_dir_edit = Qt.QLineEdit()
        fm = img_dir_edit.fontMetrics()
        img_dir_edit.setMinimumWidth(fm.width(' ' * 100))
        img_dir_bn = FixedSizePushButon('...')
        layout.addWidget(lab,
                         img_path_row, label_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(img_dir_edit,
                         img_path_row, line_edit_col,
                         1, file_bn_col - line_edit_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(img_dir_bn,
                         img_path_row, file_bn_col,
                         Qt.Qt.AlignLeft)

        # version selection
        lab = Qt.QLabel('Version :')
        version_cbx = Qt.QComboBox()
        version_cbx.addItem('0')
        version_cbx.addItem('1')
        version_cbx.setCurrentIndex(1)
        layout.addWidget(lab,
                         version_row, label_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(version_cbx,
                         version_row, line_edit_col,
                         Qt.Qt.AlignLeft)
        layout.addItem(Qt.QSpacerItem(0, 0,
                                      Qt.QSizePolicy.Expanding,
                                      Qt.QSizePolicy.Expanding))

        # last row : apply button
        parse_bn = FixedSizePushButon('Parse file')
        layout.addWidget(parse_bn,
                         apply_bn_row, 0,
                         1, last_col - first_col,
                         Qt.Qt.AlignHCenter)

        # ################
        # scans + edf QGroupBox
        # ################
        scans_gbx = GroupBox("Spec + EDF")
        grp_layout = Qt.QHBoxLayout(scans_gbx)
        self.layout().addWidget(scans_gbx,
                                1, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        # ===========
        # valid scans
        # ===========
        scan_layout = Qt.QGridLayout()
        grp_layout.addLayout(scan_layout)

        h_layout = Qt.QHBoxLayout()
        label = Qt.QLabel('<span style=" font-weight:600; color:#00916a;">'
                          'Matched scans</span>')
        label.setTextFormat(Qt.Qt.RichText)
        edit_scans_bn = FixedSizePushButon('Edit')
        h_layout.addWidget(label)
        h_layout.addWidget(edit_scans_bn)
        scan_layout.addLayout(h_layout, 0, 0, 1, 2)

        label = Qt.QLabel('Total :')
        total_scans_edit = Qt.QLineEdit('0')
        total_scans_edit.setReadOnly(True)
        fm = total_scans_edit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        total_scans_edit.setMaximumWidth(width)
        total_scans_edit.setAlignment(Qt.Qt.AlignRight)

        scan_layout.addWidget(label, 1, 0, Qt.Qt.AlignLeft)
        scan_layout.addWidget(total_scans_edit, 1, 1, Qt.Qt.AlignLeft)

        # ====

        label = Qt.QLabel('Selected :')
        selected_scans_edit = Qt.QLineEdit('0')
        selected_scans_edit.setReadOnly(True)
        fm = selected_scans_edit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        selected_scans_edit.setMaximumWidth(width)
        selected_scans_edit.setAlignment(Qt.Qt.AlignRight)

        scan_layout.addWidget(label, 2, 0, Qt.Qt.AlignLeft)
        scan_layout.addWidget(selected_scans_edit, 2, 1, Qt.Qt.AlignLeft)

        # ===

        v_line = Qt.QFrame()
        v_line.setFrameShape(Qt.QFrame.VLine)
        v_line.setFrameShadow(Qt.QFrame.Sunken)
        grp_layout.addWidget(v_line)

        # ===========
        # "other" scans
        # ===========

        scan_layout = Qt.QGridLayout()
        grp_layout.addLayout(scan_layout)

        h_layout = Qt.QHBoxLayout()
        label = Qt.QLabel('<span style=" font-weight:600; color:#ff6600;">'
                          'Other scans</span>')
        other_scans_bn = FixedSizePushButon('View')
        h_layout.addWidget(label)
        h_layout.addWidget(other_scans_bn)

        scan_layout.addLayout(h_layout, 0, 0, 1, 2)

        label = Qt.QLabel('No match :')
        no_match_scans_edit = Qt.QLineEdit('0')
        no_match_scans_edit.setReadOnly(True)
        fm = no_match_scans_edit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        no_match_scans_edit.setMaximumWidth(width)
        no_match_scans_edit.setAlignment(Qt.Qt.AlignRight)

        scan_layout.addWidget(label, 1, 0, Qt.Qt.AlignLeft)
        scan_layout.addWidget(no_match_scans_edit, 1, 1, Qt.Qt.AlignLeft)

        # ====

        label = Qt.QLabel('No img info :')
        no_img_info_edit = Qt.QLineEdit('0')
        no_img_info_edit.setReadOnly(True)
        fm = no_img_info_edit.fontMetrics()
        width = (fm.boundingRect('0123456').width() +
                 fm.boundingRect('00').width())
        no_img_info_edit.setMaximumWidth(width)
        no_img_info_edit.setAlignment(Qt.Qt.AlignRight)

        scan_layout.addWidget(label, 2, 0, Qt.Qt.AlignLeft)
        scan_layout.addWidget(no_img_info_edit, 2, 1, Qt.Qt.AlignLeft)

        # ===

        v_line = Qt.QFrame()
        v_line.setFrameShape(Qt.QFrame.VLine)
        v_line.setFrameShadow(Qt.QFrame.Sunken)
        grp_layout.addWidget(v_line)

        # ################
        # parameters
        # ################
        params_gbx = GroupBox("Acq. Parameters")
        grp_layout = Qt.QVBoxLayout(params_gbx)

        acq_params_wid = AcqParamsWidget()
        self.layout().addWidget(params_gbx,
                                2, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        grp_layout.addWidget(acq_params_wid)

        # ################
        # output options
        # ################

        output_gbx = GroupBox("Output")
        layout = Qt.QGridLayout(output_gbx)
        self.layout().addWidget(output_gbx,
                                3, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        # ===========
        # master
        # ===========

        lab = Qt.QLabel('Prefix :')
        master_edit = Qt.QLineEdit()
        fm = master_edit.fontMetrics()
        master_edit.setMinimumWidth(fm.width(' ' * 50))
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, 0, 1, Qt.Qt.AlignLeft)
        reset_bn = FixedSizePushButon('R')
        layout.addWidget(lab,
                         0, 0,
                         Qt.Qt.AlignLeft)
        sp = master_edit.sizePolicy()
        sp.setHorizontalPolicy(Qt.QSizePolicy.Maximum)
        master_edit.setSizePolicy(sp)
        h_layout.addWidget(master_edit, Qt.Qt.AlignLeft)
        h_layout.addWidget(reset_bn, Qt.Qt.AlignLeft)

        # ===========
        # output folder
        # ===========

        lab = Qt.QLabel('Output directory :')
        outdir_edit = Qt.QLineEdit()
        fm = outdir_edit.fontMetrics()
        outdir_edit.setMinimumWidth(fm.width(' ' * 100))
        outdir_bn = FixedSizePushButon('...')
        layout.addWidget(lab,
                         1, 0,
                         Qt.Qt.AlignLeft)
        layout.addWidget(outdir_edit,
                         1, 1,
                         Qt.Qt.AlignLeft)
        layout.addWidget(outdir_bn,
                         1, 2,
                         Qt.Qt.AlignLeft)

        # ################
        # merge button
        # ################

        merge_bn = Qt.QPushButton('Merge')
        cancel_bn = Qt.QPushButton('Cancel')
        h_layout = Qt.QHBoxLayout()
        self.layout().addLayout(h_layout,
                                4, 0,
                                1, 1,
                                Qt.Qt.AlignHCenter | Qt.Qt.AlignTop)
        h_layout.addWidget(merge_bn)
        h_layout.addWidget(cancel_bn)

        # #################
        # setting initial state
        # #################

        scans_gbx.setEnabled(False)
        params_gbx.setEnabled(False)
        output_gbx.setEnabled(False)
        merge_bn.setEnabled(False)
        parse_bn.setEnabled(False)

        self.__merger = None
        self.info_wid = None

        # named tuple with references to all the important widgets
        SelfWidgets = namedtuple('SelfWidgets',
                                 ['spec_file_edit',
                                  'spec_file_bn',
                                  'img_dir_edit',
                                  'img_dir_bn',
                                  'version_cbx',
                                  'parse_bn',
                                  'total_scans_edit',
                                  'selected_scans_edit',
                                  'other_scans_bn',
                                  'no_match_scans_edit',
                                  'no_img_info_edit',
                                  'acq_params_wid',
                                  'master_edit',
                                  'outdir_edit',
                                  'outdir_bn',
                                  'input_gbx',
                                  'scans_gbx',
                                  'params_gbx',
                                  'output_gbx',
                                  'merge_bn',
                                  'cancel_bn'])

        self.__widgets = SelfWidgets(spec_file_edit=spec_file_edit,
                                     spec_file_bn=spec_file_bn,
                                     img_dir_edit=img_dir_edit,
                                     img_dir_bn=img_dir_bn,
                                     version_cbx=version_cbx,
                                     parse_bn=parse_bn,
                                     total_scans_edit=total_scans_edit,
                                     selected_scans_edit=selected_scans_edit,
                                     other_scans_bn=other_scans_bn,
                                     no_match_scans_edit=no_match_scans_edit,
                                     no_img_info_edit=no_img_info_edit,
                                     acq_params_wid=acq_params_wid,
                                     master_edit=master_edit,
                                     outdir_edit=outdir_edit,
                                     outdir_bn=outdir_bn,
                                     input_gbx=input_gbx,
                                     scans_gbx=scans_gbx,
                                     params_gbx=params_gbx,
                                     output_gbx=output_gbx,
                                     merge_bn=merge_bn,
                                     cancel_bn=cancel_bn)

        self.__resetState()

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

        self.__sigParseDone.connect(self.__parseSpecDone)

        spec_file_edit.textChanged.connect(self.__inputEditTextChanged)
        spec_file_bn.clicked.connect(self.__pickSpecfile)
        img_dir_edit.textChanged.connect(self.__inputEditTextChanged)
        img_dir_bn.clicked.connect(self.__pickImageDir)
        reset_bn.clicked.connect(self.__resetMaster)
        parse_bn.clicked.connect(self.__parseSpecStart,
                                 Qt.Qt.QueuedConnection)
        outdir_edit.textChanged.connect(self.__updateOutputGroupBox)
        outdir_bn.clicked.connect(self.__pickOutputDir)
        master_edit.textChanged.connect(self.__updateOutputGroupBox)
        merge_bn.clicked.connect(self.__mergeButtonClicked)
        cancel_bn.clicked.connect(self.reject)

        edit_scans_bn.clicked.connect(self.__editScans)
        other_scans_bn.clicked.connect(self.__viewOtherScans)

        if img_dir is not None:
            img_dir_edit.setText(img_dir)
        if spec_version is not None:
            version_cbx.setCurrentIndex(spec_version)
        if output_dir is not None:
            outdir_edit.setText(output_dir)
        if spec_file is not None:
            spec_file_edit.setText(spec_file)
        self.__widget_setup = False

        self.__xsocs_h5 = None

    def showEvent(self, *args, **kwargs):
        if not self.__widget_setup:
            self.__widget_setup = True
            spec_file = self.__widgets.spec_file_edit.text()
            if len(spec_file) != 0:
                self.__widgets.parse_bn.clicked.emit(True)
        super(MergeWidget, self).showEvent(*args, **kwargs)

    def closeEvent(self, event):
        if self.__delete_tmp_root and os.path.exists(self.__tmp_root):
            shutil.rmtree(self.__tmp_root, ignore_errors=True)
        elif os.path.exists(self.__tmp_dir_merge):
            shutil.rmtree(self.__tmp_dir_merge, ignore_errors=True)
        if self.__q_tmp_dir is not None:
            # for some reason the QTemporaryDir gets deleted even thos
            # this instance is still in scope. This is a workaround until
            # we figure out what's going on.
            # (deletion seems to occur when creating the Pool in the
            # _MergeThread::run method)
            self.__q_tmp_dir.setAutoRemove(True)
        super(MergeWidget, self).closeEvent(event)

    def __resetMaster(self, *args, **kwargs):
        widgets = self.__widgets
        merger = self.__merger
        if merger is None:
            return
        merger.prefix = None
        master = merger.prefix
        widgets.master_edit.setText(master)

    def __mergeButtonClicked(self, *args, **kwargs):
        widgets = self.__widgets
        merger = self.__merger

        if merger is None:
            # this part shouldn't even be called, just putting this
            # in case someone decides to modify the code to enable the merge_bn
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

        acq_params_wid = widgets.acq_params_wid
        try:
            name = 'Beam Energy'
            merger.beam_energy = \
                assert_non_none(acq_params_wid.beam_energy)

            name = 'Direct beam'
            dir_beam_h = assert_non_none(acq_params_wid.direct_beam_h)
            dir_beam_v = assert_non_none(acq_params_wid.direct_beam_v)
            merger.center_chan = [dir_beam_h, dir_beam_v]

            name = 'Channel per degree'
            chpdeg_h = assert_non_none(acq_params_wid.chperdeg_h)
            chpdeg_v = assert_non_none(acq_params_wid.chperdeg_v)
            merger.chan_per_deg = [chpdeg_h, chpdeg_v]

            name = 'Prefix'
            master = str(widgets.master_edit.text())
            if len(master) == 0:
                raise ValueError('parameter is mandatory.')
            merger.prefix = master

            name = 'output_dir'
            output_dir = str(widgets.outdir_edit.text())
            if len(output_dir) == 0:
                raise ValueError('parameter is mandatory.')
            merger.output_dir = output_dir

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
        process_diag.accepted.connect(partial(self.__mergeDone, status=True))
        process_diag.rejected.connect(partial(self.__mergeDone, status=False))
        process_diag.setModal(True)
        self.__process_diag = process_diag
        process_diag.show()

    def __mergeDone(self, status):
        self.__process_diag = None
        if status:
            self.__xsocs_h5 = self.__merger.master_file
            self.accept()

    @property
    def xsocsH5(self):
        return self.__xsocs_h5

    def __inputEditTextChanged(self, text):
        """
        Checking if the provided paths exist.
        If not, disable the parse button.
        """
        widgets = self.__widgets
        sender = self.sender()

        enabled = False

        if sender == widgets.spec_file_edit:
            if os.path.isfile(text):
                enabled = True
            else:
                enabled = False

        elif sender == widgets.img_dir_edit:
            if len(text) == 0 or os.path.isdir(text):
                enabled = True
            else:
                enabled = False

        widgets.parse_bn.setEnabled(enabled)

        self.__resetState()

    def __resetState(self):
        """
        Sets the default state for the groupboxes
        - disables all but the input groupbox
        - clears the scan widget
        """
        widgets = self.__widgets
        widgets.scans_gbx.setEnabled(False)
        widgets.params_gbx.setEnabled(False)
        widgets.output_gbx.setEnabled(False)
        self.__merger = None

    def __pickSpecfile(self, *args, **kwargs):
        """
        Spec file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select SPEC file',
                                filter=('Scan files (*.spec);;'
                                        'Any files (*)'))
        dialog.setFileMode(Qt.QFileDialog.ExistingFile)
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            widgets.spec_file_edit.setText(file_name)

    def __pickImageDir(self, *args, **kwargs):
        """
        Img dir file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select EDF folder')
        dialog.setFileMode(Qt.QFileDialog.Directory)
        if dialog.exec_():
            dir_name = dialog.selectedFiles()[0]
            widgets.img_dir_edit.setText(dir_name)

    def __pickOutputDir(self, *args, **kwargs):
        """
        Output dir file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select output directory')
        dialog.setFileMode(Qt.QFileDialog.Directory)
        if dialog.exec_():
            dir_name = dialog.selectedFiles()[0]
            widgets.outdir_edit.setText(dir_name)

        self.__updateOutputGroupBox()

    def __editScans(self, *args, **kwargs):
        merger = self.__merger
        if merger is None:
            return
        ans = _ScansSelectDialog(merger, parent=self).exec_()
        if ans == Qt.QDialog.Accepted:
            self.__updateScansInfos()

    def __viewOtherScans(self, *args, **kwargs):
        merger = self.__merger
        if merger is None:
            return
        _ScansInfoDialog(merger, parent=self).exec_()

    def __updateOutputGroupBox(self, first=False, **kwargs):
        widgets = self.__widgets
        merger = self.__merger

        if merger is None:
            enable = False
        else:
            enable = merger.parsed and len(merger.matched_ids) > 0

        widgets.output_gbx.setEnabled(enable)

        if enable:
            # improve on this
            if first:
                #master = merger.master_file
                prefix = widgets.master_edit.text()
                if len(prefix) == 0:
                    widgets.master_edit.setText(merger.prefix)

            has_output_dir = len(widgets.outdir_edit.text()) > 0
            has_prefix = len(widgets.master_edit.text()) > 0
            enable = has_output_dir and has_prefix
            widgets.merge_bn.setEnabled(enable)
        else:
            widgets.master_edit.clear()

    def __parseSpecStart(self, *args, **kwargs):
        self.info_wid = None

        widgets = self.__widgets

        version = widgets.version_cbx.currentIndex()
        spec_file = widgets.spec_file_edit.text()
        img_dir = widgets.img_dir_edit.text()

        if len(img_dir) == 0:
            img_dir = None
        else:
            img_dir = str(img_dir)

        try:
            if self.__merger is None:
                self.__merger = Id01DataMerger(str(spec_file),
                                               self.__tmp_dir_merge,
                                               img_dir=img_dir,
                                               version=version)
            else:
                self.__merger.reset(str(spec_file),
                                    img_dir=img_dir,
                                    version=version)
        except Exception as ex:
            msg = ('Parsing failed: {0}.\n'
                   'Message : {1}.'
                   ''.format(ex.__class__.__name__, str(ex)))
            Qt.QMessageBox.critical(self,
                                    'Parse error.',
                                    msg)
            return

        widgets.parse_bn.setEnabled(False)
        widgets.parse_bn.setText('Parsing...')

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
        self.__merger.parse(blocking=False,
                            callback=self.__sigParseDone.emit)

    def __parseSpecDone(self, *args, **kwargs):
        widgets = self.__widgets

        self.info_wid.done(0)
        self.info_wid = None

        widgets.parse_bn.setEnabled(True)
        widgets.parse_bn.setText('Parse file.')

        self.__updateScansInfos()
        self.__updateOutputGroupBox(first=True)

    def __updateScansInfos(self):
        """
        Fills the scans group box with the results
        of the scan.
        """
        widgets = self.__widgets
        merger = self.__merger

        if merger is None:
            matched_ids = []
            selected_ids = []
            no_match_ids = []
            no_img_ids = []
            enable = False
        else:
            matched_ids = merger.matched_ids
            selected_ids = merger.selected_ids
            no_match_ids = merger.no_match_ids
            no_img_ids = merger.no_img_ids
            enable = merger.parsed

        n_total = len(matched_ids)
        n_selected = len(selected_ids)
        n_no_match = len(no_match_ids)
        n_no_img = len(no_img_ids)

        widgets.scans_gbx.setEnabled(enable)
        widgets.params_gbx.setEnabled(len(matched_ids) > 0)

        widgets.total_scans_edit.setText(str(n_total))
        widgets.selected_scans_edit.setText(str(n_selected))
        widgets.no_match_scans_edit.setText(str(n_no_match))
        widgets.no_img_info_edit.setText(str(n_no_img))


if __name__ == '__main__':
    pass
