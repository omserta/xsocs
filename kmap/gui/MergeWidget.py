#!/users/naudet/bin/python_id01

import os
import sys
import time
import shutil
import threading
from collections import namedtuple

import kmap as xsocs
from kmap.util.id01_spec import Id01DataMerger

from silx.gui import qt as Qt

_MAX_BEAM_ENERGY_EV = 10**6

_MU_LOWER = u'\u03BC'
_PHI_LOWER = u'\u03C6'

_HELP_WIDGET_STYLE = """
            QLabel {
                border-radius: 10px;
                padding: 1px 4px;
                background-color: qradialgradient(spread:reflect, cx:0.5, cy:0.5, radius:1, fx:0.5, fy:0.5, stop:0 rgba(0, 0, 255, 255), stop:1 rgba(255, 255, 255, 255));
                color: rgb(255, 255, 255);
            }"""


def _create_tmp_dir():
    
    qt_tmp_tpl = os.path.join(Qt.QDir.tempPath(),
                               'tmpXsocsXXXXXX')
    try:
        tmp_dir = Qt.QTemporaryDir(qt_tmp_tpl)
        isValid = tmp_dir.isValid()
        delete_tmp = False
    except AttributeError:
        isValid = False

    if not isValid:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        delete_tmp = True

    return tmp_dir, delete_tmp

class _AdjustedPushButton(Qt.QPushButton):
    """
    It seems that by default QPushButtons minimum width is 75.
    This is a workaround.
    For _AdjustedPushButton to work text has to be set at creation time.
    """
    def __init__(self, text, padding=None, **kwargs):
        super(_AdjustedPushButton, self).__init__(text, **kwargs)

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        width = fm.width(self.text()) + padding
        self.setMaximumWidth(width)


class _AdjustedLineEdit(Qt.QLineEdit):
    """
    """
    def __init__(self, width, padding=None, **kwargs):
        super(_AdjustedLineEdit, self).__init__(**kwargs)

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        text = '0' * width
        width = fm.width(text) + padding
        self.setMaximumWidth(width)


class _SpinBoxLayout(Qt.QHBoxLayout):
    def __init__(self,
                 klass=Qt.QDoubleSpinBox,
                 unit=None,
                 min_value=None,
                 max_value=None,
                 special_val_txt=None,
                 **kwargs):
        super(_SpinBoxLayout, self).__init__(**kwargs)
        spinbox = self.__spinbox = klass()
        label = self.__label = Qt.QLabel(unit)

        if min_value is not None:
            spinbox.setMinimum(min_value)
        if max_value is not None:
            spinbox.setMaximum(max_value)
        if special_val_txt is not None:
            spinbox.setSpecialValueText(special_val_txt)

        self.addWidget(self.__spinbox)
        self.addWidget(self.__label)

    @property
    def spinbox(self):
        return self.__spinbox

    @property
    def label(self):
        return self.__label


class _ScansSelectDialog(Qt.QDialog):
    merge_done = Qt.Signal()
    
    def __init__(self, merger, **kwargs):
        super(_ScansSelectDialog, self).__init__(**kwargs)
        layout = Qt.QVBoxLayout(self)

        matched = merger.matched_ids
        selected = merger.selected_ids

        table_widget = Qt.QTableWidget(len(matched), 3)

        for num, scan_id in enumerate(matched):
            item = Qt.QTableWidgetItem()
            item.setFlags(Qt.Qt.ItemIsUserCheckable |
                            Qt.Qt.ItemIsEditable |
                            Qt.Qt.ItemIsSelectable |
                            Qt.Qt.ItemIsEnabled)
            state = Qt.Qt.Checked if scan_id in selected else Qt.Qt.Unchecked
            item.setCheckState(state)
            table_widget.setItem(num, 0, item)

            item = Qt.QTableWidgetItem(str(scan_id))
            table_widget.setItem(num, 1, item)

            img_file = merger.get_scan_image(scan_id)
            item = Qt.QTableWidgetItem(img_file)
            item.setToolTip(img_file)
            table_widget.setItem(num, 2, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()

        layout.addWidget(table_widget)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Apply')

        layout.addWidget(bn_box)
        bn_box.accepted.connect(self.__on_accept)
        bn_box.rejected.connect(self.reject)

        self.__table_widget = table_widget
        self.__merger = merger

    def __on_accept(self, *args, **kwags):
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
    merge_done = Qt.Signal()

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
            table_widget.setItem(num, 1, item)

        offset = len(no_match)

        for num, scan_id in enumerate(no_img):
            item = Qt.QTableWidgetItem(scan_id)
            table_widget.setItem(num + offset, 0, item)

            item = Qt.QTableWidgetItem('No image info in header.')
            table_widget.setItem(num + offset, 1, item)

        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        table_widget.sortByColumn(0, Qt.Qt.AscendingOrder)

        layout.addWidget(table_widget)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Close)

        layout.addWidget(bn_box)
        bn_box.rejected.connect(self.reject)

class _MergeProcessDialog(Qt.QDialog):
    merge_done = Qt.Signal()
    
    def __init__(self, merge_worker, **kwargs):
        super(_MergeProcessDialog, self).__init__(**kwargs)
        layout = Qt.QVBoxLayout(self)

        merger = merge_worker.merger

        files = merge_worker.merger.summary()
        output_dir = merger.output_dir

        label = Qt.QLabel('<html><head/><body><p align="center">'
                          '<span style=" font-size:16pt; font-weight:600;">'
                          'Merge process</span></p></body></html>')
        label.setTextFormat(Qt.Qt.RichText)
        layout.addWidget(label, stretch=0, alignment=Qt.Qt.AlignHCenter)

        grp_box = Qt.QGroupBox('Output directory :')
        grp_box.setLayout(Qt.QVBoxLayout())
        outdir_edit = Qt.QLineEdit(output_dir)
        fm = outdir_edit.fontMetrics()
        outdir_edit.setMinimumWidth(fm.width(' ' * 100))
        grp_box.layout().addWidget(outdir_edit)

        layout.addWidget(grp_box, stretch=0)
        grp_box = Qt.QGroupBox('Files :')
        grp_box.setLayout(Qt.QVBoxLayout())
        tree_widget = Qt.QTreeWidget()
        tree_widget.setColumnCount(3)
        tree_widget.setColumnHidden(2, True)
        # TODO improve
        master_item = Qt.QTreeWidgetItem([files['master'], '', 'master'])
        {master_item.addChild(Qt.QTreeWidgetItem([files[scan_id], '', scan_id]))
         for scan_id in sorted(files.keys()) if scan_id != 'master'}
        tree_widget.addTopLevelItem(master_item)
        wid = Qt.QProgressBar()
        tree_widget.setItemWidget(master_item, 1, Qt.QProgressBar())
        {tree_widget.setItemWidget(master_item.child(i_child),
                                                     1,
                                                     Qt.QProgressBar())
         for i_child in range(master_item.childCount())}
        master_item.setExpanded(True)
        tree_widget.resizeColumnToContents(0)
        tree_widget.resizeColumnToContents(1)
        width = tree_widget.sizeHintForColumn(0) + tree_widget.sizeHintForColumn(1) + 10
        tree_widget.setMinimumWidth(width)
        layout.addWidget(tree_widget, stretch=1, alignment=Qt.Qt.AlignHCenter)

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Merge')
        layout.addWidget(bn_box)
        bn_box.accepted.connect(self.__on_accept)
        bn_box.rejected.connect(self.reject)

        merge_worker.merge_done.connect(self.__merge_done)
        self.__merge_worker = merge_worker
        self.__tree_widget = tree_widget
        self.__bn_box = bn_box
        self.__abort_diag = None

    def __on_accept(self, *args, **kwargs):
        merger = self.__merge_worker.merger
        files = merger.summary(fullpath=True)

        warn = any(os.path.exists(f) for f in files.values())

        if warn:
            ans = Qt.QMessageBox.warning(self,
                                        'Overwrite?',
                                        ('Some files already exist.'
                                        '\nDo you want to overwrite them?'),
                                        buttons=Qt.QMessageBox.Yes |
                                        Qt.QMessageBox.No)
            if ans == Qt.QMessageBox.No:
                return

        self.__merge_start()
    
    def __merge_start(self):
        self.__bn_box.rejected.disconnect(self.reject)
        self.__bn_box.rejected.connect(self.__on_abort)
        self.__bn_box.button(Qt.QDialogButtonBox.Ok).setEnabled(False)
        self.__bn_box.button(Qt.QDialogButtonBox.Cancel).setText('Abort')

        self.__qtimer = Qt.QTimer()
        self.__qtimer.timeout.connect(self.__on_progress)
        self.__merge_worker.merge()
        self.__on_progress()
        self.__qtimer.start(1000)

        self.__time = time.time()

    def __on_abort(self, *args, **kwargs):
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
        self.__merge_worker.merger.abort_merge(wait=False)

    def __merge_done(self):
        print 'TOTAL', time.time() - self.__time
        self.__qtimer.stop()
        self.__qtimer = None
        self.__on_progress()
        if self.__abort_diag is not None:
            self.__abort_diag.done(0)
            self.__abort_diag = None
        self.__bn_box.button(Qt.QDialogButtonBox.Cancel).setText('Close')
        self.__bn_box.rejected.connect(self.accept)

    def __on_progress(self, *args, **kwargs):
        progress = self.__merge_worker.merger.progress()
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
            if len(item)>0:
                item = item[0]
                wid = tree_wid.itemWidget(item, 1)
                wid.setValue(prog)
                #if prog > 0:
                    #if not isinstance(wid, Qt.QProgressBar):
                        #wid = Qt.QProgressBar()
                        #tree_wid.setItemWidget(item, 1, wid)
                    #wid.setValue(prog)
                #elif prog < 0:
                    #if isinstance(wid, Qt.QProgressBar):
                        ##wid = Qt.QLabel('Err')
                        #tree_wid.setItemWidget(item, 1, None)
                        #item.setText(1, 'err')
        item = tree_wid.findItems('master',
                                  flags,
                                  column=2)
        if len(item)>0:
            item = item[0]
            wid = tree_wid.itemWidget(item, 1)
            wid.setValue(total / len(progress))


class MergeWidget(Qt.QWidget):

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

        input_gbx = Qt.QGroupBox("Input")
        layout = Qt.QGridLayout(input_gbx)
        self.layout().addWidget(input_gbx,
                                0, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        first_col = 0
        label_col = first_col
        line_edit_col = 1
        file_bn_col = 4
        vers_help_col = 2
        vers_spacer_col = 3
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
        spec_file_bn = _AdjustedPushButton('...')
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
        img_dir_bn = _AdjustedPushButton('...')
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
        version_help_bn = Qt.QLabel('?')
        # TODO : use icon instead?
        version_help_bn.setStyleSheet(_HELP_WIDGET_STYLE)
        version_help_bn.setToolTip('Todo')
        layout.addWidget(lab,
                         version_row, label_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(version_cbx,
                         version_row, line_edit_col,
                         Qt.Qt.AlignLeft)
        layout.addWidget(version_help_bn,
                         version_row, vers_help_col,
                         Qt.Qt.AlignLeft)
        layout.addItem(Qt.QSpacerItem(0, 0,
                                      Qt.QSizePolicy.Expanding,
                                      Qt.QSizePolicy.Expanding))

        # last row : apply button
        parse_bn = _AdjustedPushButton('Parse file')
        layout.addWidget(parse_bn,
                         apply_bn_row, 0,
                         1, last_col - first_col,
                         Qt.Qt.AlignHCenter)

        # ################
        # scans + edf QGroupBox
        # ################
        scans_gbx = Qt.QGroupBox("Spec + EDF")
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
        edit_scans_bn = _AdjustedPushButton('Edit')
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
        v_line.setFrameShape(Qt.QFrame.VLine);
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
        other_scans_bn = _AdjustedPushButton('View')
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
        v_line.setFrameShape(Qt.QFrame.VLine);
        v_line.setFrameShadow(Qt.QFrame.Sunken)
        grp_layout.addWidget(v_line)

        ## ===========
        ## parsing errors scans
        ## ===========

        #scan_layout = Qt.QGridLayout()
        #grp_layout.addLayout(scan_layout)

        #h_layout = Qt.QHBoxLayout()
        #label = Qt.QLabel('<span style=" font-weight:600; color:#ff6600;">'
                          #'Invalid scans</span>')
        #invalid_scans_bn = _AdjustedPushButton('View')
        #h_layout.addWidget(label)
        #h_layout.addWidget(invalid_scans_bn)

        #scan_layout.addLayout(h_layout, 0, 0, 1, 2)

        #label = Qt.QLabel('Errors :')
        #unmatched_scans_edit = Qt.QLineEdit('0')
        #unmatched_scans_edit.setReadOnly(True)
        #fm = unmatched_scans_edit.fontMetrics()
        #width = (fm.boundingRect('0123456').width() +
                 #fm.boundingRect('00').width())
        #unmatched_scans_edit.setMaximumWidth(width)
        #unmatched_scans_edit.setAlignment(Qt.Qt.AlignRight)

        #scan_layout.addWidget(label, 1, 0, Qt.Qt.AlignLeft)
        #scan_layout.addWidget(unmatched_scans_edit, 1, 1, Qt.Qt.AlignLeft)

        # ################
        # parameters
        # ################
        params_gbx = Qt.QGroupBox("Acq. Parameters")
        grp_layout = Qt.QVBoxLayout(params_gbx)
        self.layout().addWidget(params_gbx,
                                2, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        layout = Qt.QGridLayout()
        grp_layout.addLayout(layout)

        # ===========
        # beam energy
        # ===========

        # could put the unit in the spinbox using setSuffix, but I feel like
        # it's more readable if it's outside.
        # TODO : clear button?
        row = 0
        beam_nrg_edit = _AdjustedLineEdit(8)#Qt.QLineEdit()#Qt.QDoubleSpinBox()
        beam_nrg_edit.setValidator(Qt.QDoubleValidator(beam_nrg_edit))
        #beam_nrg_spin.setSpecialValueText('N/A')
        #beam_nrg_spin.setMaximum(_MAX_BEAM_ENERGY_EV)
        #beam_nrg_spin.setMinimum(0.)
        beam_nrg_edit.setAlignment(Qt.Qt.AlignRight)
        layout.addWidget(Qt.QLabel('Beam energy :'), row, 0)
        layout.addWidget(beam_nrg_edit, row, 1, Qt.Qt.AlignRight)
        layout.addWidget(Qt.QLabel('eV'), row, 2)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine);
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # pristine beam
        # ===========
        row += 1
        h_layout = Qt.QHBoxLayout()
        v_layout = Qt.QFormLayout()
        dir_beam_h_edit = _AdjustedLineEdit(8)
        dir_beam_h_edit.setValidator(Qt.QDoubleValidator(dir_beam_h_edit))
        dir_beam_h_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('h=', dir_beam_h_edit)
        dir_beam_v_edit = _AdjustedLineEdit(8)
        dir_beam_v_edit.setValidator(Qt.QDoubleValidator(dir_beam_v_edit))
        dir_beam_v_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('v=', dir_beam_v_edit)
        h_layout.addLayout(v_layout)
        layout.addWidget(Qt.QLabel('Direct beam :'), row, 0)
        layout.addLayout(h_layout, row, 1)
        layout.addWidget(Qt.QLabel('px'), row, 2)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine);
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # chan per degree
        # ===========

        row += 1
        h_layout = Qt.QHBoxLayout()
        v_layout = Qt.QFormLayout()
        chpdeg_h_edit = _AdjustedLineEdit(8)
        chpdeg_h_edit.setValidator(Qt.QDoubleValidator(chpdeg_h_edit))
        chpdeg_h_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('h=', chpdeg_h_edit)
        chpdeg_v_edit = _AdjustedLineEdit(8)
        chpdeg_v_edit.setValidator(Qt.QDoubleValidator(chpdeg_v_edit))
        chpdeg_v_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('v=', chpdeg_v_edit)
        h_layout.addLayout(v_layout)
        layout.addWidget(Qt.QLabel('Chan. per deg. :'), row, 0)
        layout.addLayout(h_layout, row, 1)
        layout.addWidget(Qt.QLabel('px'), row, 2)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine);
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # pixelsize
        # ===========

        row += 1
        h_layout = Qt.QHBoxLayout()
        v_layout = Qt.QFormLayout()
        pixelsize_h_edit = _AdjustedLineEdit(8)
        pixelsize_h_edit.setValidator(Qt.QDoubleValidator(pixelsize_h_edit))
        pixelsize_h_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('h=', pixelsize_h_edit)
        pixelsize_v_edit = _AdjustedLineEdit(8)
        pixelsize_v_edit.setValidator(Qt.QDoubleValidator(pixelsize_v_edit))
        pixelsize_v_edit.setAlignment(Qt.Qt.AlignRight)
        v_layout.addRow('v=', pixelsize_v_edit)
        h_layout.addLayout(v_layout)
        layout.addWidget(Qt.QLabel('Pixel size :'), row, 0)
        layout.addLayout(h_layout, row, 1)
        layout.addWidget(Qt.QLabel('TBD'), row, 2)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine);
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # detector orientation
        # ===========

        row += 1
        layout.addWidget(Qt.QLabel('Det. orientation :'), row, 0)
        h_layout = Qt.QHBoxLayout()
        v_layout = Qt.QFormLayout()
        h_layout.addLayout(v_layout)
        det_phi_rb = Qt.QRadioButton(u'Width is {0}.'.format(_PHI_LOWER))
        v_layout.addRow(det_phi_rb)
        det_mu_rb = Qt.QRadioButton(u'Width is {0}.'.format(_MU_LOWER))
        v_layout.addRow(det_mu_rb)
        layout.addLayout(h_layout, row, 1)

        # ################
        # output options
        # ################

        output_gbx = Qt.QGroupBox("Output")
        layout = Qt.QGridLayout(output_gbx)
        self.layout().addWidget(output_gbx,
                                3, 0,
                                Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        # ===========
        # master
        # ===========

        lab = Qt.QLabel('Master file :')
        master_edit = Qt.QLineEdit()
        fm = master_edit.fontMetrics()
        master_edit.setMinimumWidth(fm.width(' ' * 50))
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, 0, 1, Qt.Qt.AlignLeft)
        reset_bn = _AdjustedPushButton('R')
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
        outdir_bn = _AdjustedPushButton('...')
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

        self.__merge_worker = _MergeWorker()
        self.__merge_worker.parse_done.connect(self.__parse_spec_done)
        self.info_wid = None

        # named tuple with references to all the important widgets
        SelfWidgets = namedtuple('SelfWidgets',
                                 ['spec_file_edit',
                                  'spec_file_bn',
                                  'img_dir_edit',
                                  'img_dir_bn',
                                  'version_cbx',
                                  'version_help_bn',
                                  'parse_bn',
                                  'total_scans_edit',
                                  'selected_scans_edit',
                                  'other_scans_bn',
                                  'no_match_scans_edit',
                                  'no_img_info_edit',
                                  'beam_nrg_edit',
                                  'dir_beam_h_edit',
                                  'dir_beam_v_edit',
                                  'chpdeg_h_edit',
                                  'chpdeg_v_edit',
                                  'pixelsize_h_edit',
                                  'pixelsize_v_edit',
                                  'det_phi_rb',
                                  'det_mu_rb',
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
                                     version_help_bn=version_help_bn,
                                     parse_bn=parse_bn,
                                     total_scans_edit=total_scans_edit,
                                     selected_scans_edit=selected_scans_edit,
                                     other_scans_bn=other_scans_bn,
                                     no_match_scans_edit=no_match_scans_edit,
                                     no_img_info_edit=no_img_info_edit,
                                     beam_nrg_edit=beam_nrg_edit,
                                     dir_beam_h_edit=dir_beam_h_edit,
                                     dir_beam_v_edit=dir_beam_v_edit,
                                     chpdeg_h_edit=chpdeg_h_edit,
                                     chpdeg_v_edit=chpdeg_v_edit,
                                     pixelsize_h_edit=pixelsize_h_edit,
                                     pixelsize_v_edit=pixelsize_v_edit,
                                     det_phi_rb=det_phi_rb,
                                     det_mu_rb=det_mu_rb,
                                     master_edit=master_edit,
                                     outdir_edit=outdir_edit,
                                     outdir_bn=outdir_bn,
                                     input_gbx=input_gbx,
                                     scans_gbx=scans_gbx,
                                     params_gbx=params_gbx,
                                     output_gbx=output_gbx,
                                     merge_bn=merge_bn,
                                     cancel_bn=cancel_bn)

        self.__reset_state()

        if tmp_dir is None:
            tmp_dir, delete_tmp = _create_tmp_dir()
        else:
            delete_tmp = False

        self.__tmp_root = tmp_dir
        self.__delete_tmp_root = delete_tmp

        tmp_dir = os.path.join(self.__tmp_root, 'xsocs_merge')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self.__tmp_dir_merge = tmp_dir

        print('Using temporary fold : {0}.'.format(tmp_dir))

        spec_file_edit.textChanged.connect(self.__input_edit_textChanged)
        spec_file_bn.clicked.connect(self.__pick_specfile)
        img_dir_edit.textChanged.connect(self.__input_edit_textChanged)
        img_dir_bn.clicked.connect(self.__pick_imgdir)
        reset_bn.clicked.connect(self.__reset_master)
        parse_bn.clicked.connect(self.__parse_spec_start,
                                 Qt.Qt.QueuedConnection)
        outdir_edit.textChanged.connect(self.__update_output_gbx)
        outdir_bn.clicked.connect(self.__pick_output_dir)
        master_edit.textChanged.connect(self.__update_output_gbx)
        master_edit.editingFinished.connect(self.__master_edit_editingFinished)
        merge_bn.clicked.connect(self.__merge_bn_clicked)
        cancel_bn.clicked.connect(self.close)

        edit_scans_bn.clicked.connect(self.__edit_scans)
        other_scans_bn.clicked.connect(self.__view_other_scans)

        if img_dir is not None:
            img_dir_edit.setText(img_dir)
        if spec_version is not None:
            version_cbx.setCurrentIndex(spec_version)
        if output_dir is not None:
            outdir_edit.setText(output_dir)
        if spec_file is not None:
            spec_file_edit.setText(spec_file)
        self.__widget_setup = False

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
        if self.__merge_worker:
            self.__merge_worker.exit(0)
        super(MergeWidget, self).closeEvent(event)

    def __reset_master(self, *args, **kwargs):
        widgets = self.__widgets
        merger = self.__merge_worker.merger
        if merger is None:
            return
        merger.set_master_file(None)
        master = merger.master_file
        widgets.master_edit.setText(master)

    def __merge_bn_clicked(self, *args, **kwargs):
        widgets = self.__widgets
        merge_worker = self.__merge_worker
        merger = self.__merge_worker.merger

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

        to_double_or_none = lambda txt: None if len(txt) == 0 else float(txt)
        try:
            name = 'Beam Energy'
            beam_energy = to_double_or_none(widgets.beam_nrg_edit.text())

            name = 'Direct beam'
            dir_beam_h = to_double_or_none(widgets.dir_beam_h_edit.text())
            dir_beam_v = to_double_or_none(widgets.dir_beam_v_edit.text())
            if (dir_beam_h is None) ^ (dir_beam_v is None):
                raise ValueError('both values must be set (or none of them).')
            center_chan = [dir_beam_h, dir_beam_v]
            if None in center_chan:
                center_chan = None

            name = 'Channel per degree'
            chpdeg_h = to_double_or_none(widgets.chpdeg_h_edit.text())
            chpdeg_v = to_double_or_none(widgets.chpdeg_v_edit.text())
            if (chpdeg_h is None) ^ (chpdeg_v is None):
                raise ValueError('both values must be set (or none of them).')
            chan_per_deg = [chpdeg_h, chpdeg_v]
            if None in chan_per_deg:
                chan_per_deg = None

            name = 'Pixe size'
            pixelsize_h = to_double_or_none(widgets.pixelsize_h_edit.text())
            pixelsize_v = to_double_or_none(widgets.pixelsize_v_edit.text())
            if (pixelsize_h is None) ^ (pixelsize_v is None):
                raise ValueError('both values must be set (or none of them).')
            pixelsize = [pixelsize_h, pixelsize_v]
            if None in pixelsize:
                pixelsize = None

            name = 'Detector orientation'
            if widgets.det_mu_rb.isChecked():
                det_orientation = 'mu'
            elif widgets.det_phi_rb.isChecked():
                det_orientation = 'phi'
            else:
                det_orientation = None

            name = 'master'
            master = widgets.master_edit.text()
            if len(master) == 0:
                raise ValueError('field is mandatory.')

            output_dir = widgets.outdir_edit.text()
            
            merger.set_output_dir(str(output_dir))
            merger.beam_energy = beam_energy
            merger.center_chan = center_chan
            merger.chan_per_deg = chan_per_deg
            merger.pixelsize = pixelsize
            merger.detector_orient = det_orientation
        except Exception as ex:
            Qt.QMessageBox.critical(self, 'Error',
                                    '{0} : {1}.'.format(name, str(ex)))
            return

        process_diag = _MergeProcessDialog(merge_worker, parent=self)
        process_diag.setAttribute(Qt.Qt.WA_DeleteOnClose)
        process_diag.accepted.connect(self.__merge_done)
        process_diag.rejected.connect(self.__merge_done)
        self.__process_diag = process_diag
        process_diag.show()

    def __merge_done(self, *args, **kwargs):
        #self.__process_diag.done(0)
        self.__process_diag = None
        print('TODO : DONE')

    def __input_edit_textChanged(self, text):
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
            if len(text)==0 or os.path.isdir(text):
                enabled = True
            else:
                enabled = False

        widgets.parse_bn.setEnabled(enabled)

        self.__reset_state()

    def __reset_state(self):
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

    def __pick_specfile(self, *args, **kwargs):
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
            file_name = dialog.selectedFiles()[0];
            widgets.spec_file_edit.setText(file_name)

    def __pick_imgdir(self, *args, **kwargs):
        """
        Img dir file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select EDF folder')
        dialog.setFileMode(Qt.QFileDialog.Directory)
        if dialog.exec_():
            dir_name = dialog.selectedFiles()[0];
            widgets.img_dir_edit.setText(dir_name)

    def __pick_output_dir(self, *args, **kwargs):
        """
        Output dir file picker
        """
        widgets = self.__widgets
        dialog = Qt.QFileDialog(self,
                                'Select output directory')
        dialog.setFileMode(Qt.QFileDialog.Directory)
        if dialog.exec_():
            dir_name = dialog.selectedFiles()[0];
            widgets.outdir_edit.setText(dir_name)
        
        self.__update_output_gbx()

    def __edit_scans(self, *args, **kwargs):
        merger = self.__merge_worker.merger
        if merger is None:
            return
        ans = _ScansSelectDialog(merger, parent=self).exec_()
        if ans == Qt.QDialog.Accepted:
            self.__update_scans_infos()

    def __view_other_scans(self, *args, **kwargs):
        merger = self.__merge_worker.merger
        if merger is None:
            return
        ans = _ScansInfoDialog(merger, parent=self).exec_()

    def __master_edit_editingFinished(self, *args, **kwargs):
        widgets = self.__widgets
        merger = self.__merger
        line_edit = self.sender()
        
        if merger is None:
            return
        merger.set_master_file(str(line_edit.text()))
        master = merger.master_file
        line_edit.setText(master)

    def __update_output_gbx(self, *args, **kwargs):
        widgets = self.__widgets
        merger = self.__merger

        if merger is None:
            enable = False
        else:
            selected_ids = merger.selected_ids
            if len(selected_ids) > 0:
                enable = True
            else:
                enable = False
        widgets.output_gbx.setEnabled(enable)
        
        if enable:
            selected = merger.selected_ids[0]
            master = merger.master_file
            # improve on this
            master = widgets.master_edit.text()
            if len(master) == 0:
                widgets.master_edit.setText(merger.master_file)

            has_output_dir = len(widgets.outdir_edit.text()) > 0
            has_master = len(widgets.master_edit.text()) > 0
            enable = has_output_dir and has_master
            widgets.merge_bn.setEnabled(enable)
        else:
            widgets.master_edit.clear()

    def __parse_spec_start(self, *args, **kwargs):
        self.info_wid = None

        widgets = self.__widgets

        version = widgets.version_cbx.currentIndex()
        spec_file = widgets.spec_file_edit.text()
        img_file = widgets.img_dir_edit.text()

        if len(img_file) == 0:
            img_file = None
        else:
            img_file = str(img_file)

        try:
            id01_merger = Id01DataMerger(str(spec_file),
                                         self.__tmp_dir_merge,
                                         img_dir_base=img_file,
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

        self.__merger = id01_merger
        self.__merge_worker.setMerger(id01_merger)
        self.__merge_worker.parse()

    def __parse_spec_done(self, *args, **kwargs):
        widgets = self.__widgets

        self.info_wid.done(0)
        self.info_wid = None

        widgets.parse_bn.setEnabled(True)
        widgets.parse_bn.setText('Parse file.')

        self.__update_scans_infos()
        self.__update_output_gbx()

    def __update_scans_infos(self):
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
        else:
            matched_ids = merger.matched_ids
            selected_ids = merger.selected_ids
            no_match_ids = merger.no_match_ids
            no_img_ids = merger.no_img_ids

        n_total = len(matched_ids)
        n_selected = len(selected_ids)
        n_no_match = len(no_match_ids)
        n_no_img = len(no_img_ids)
        
        if n_selected > 0:
            widgets.scans_gbx.setEnabled(True)
            widgets.params_gbx.setEnabled(True)
        else:
            widgets.scans_gbx.setEnabled(False)
            widgets.params_gbx.setEnabled(False)

        widgets.total_scans_edit.setText(str(n_total))
        widgets.selected_scans_edit.setText(str(n_selected))
        widgets.no_match_scans_edit.setText(str(n_no_match))
        widgets.no_img_info_edit.setText(str(n_no_img))


def before_after(f):  
    def decorator(*args, **kwargs):  
        print('before', f.func_name)  
        f(*args, **kwargs)  
        print('after', f.func_name)  
    return decorator  

class _MergeWorker(Qt.QObject):
    """
    WARNING : NOT thread safe!!
    """
    parse_done = Qt.Signal()
    merge_done = Qt.Signal()
    parse_started = Qt.Signal()
    merge_started = Qt.Signal()

    def __init__(self,
                 merger_obj=None):
        super(_MergeWorker, self).__init__()
        self.__thread = Qt.QThread(self)

        self.merger = merger_obj

        self.moveToThread(self.__thread)
        
        self.__thread.start()

        self.parse_started.connect(self.__do_parse)
        self.merge_started.connect(self.__do_merge)

    def setMerger(self, merger):
        self.merger = merger

    def parse(self):
        if self.merger is None:
            raise ValueError('No merger set.')
        self.parse_started.emit()

    def merge(self):
        if self.merger is None:
            raise ValueError('No merger set.')
        self.merge_started.emit()

    def __do_parse(self):
        self.merger.parse()
        self.parse_done.emit()

    def __do_merge(self):
        self.merger.merge(overwrite=True,
                          blocking=False,
                          callback=self.__on_done)

    def __on_done(self):
        self.merge_done.emit()

    def exit(self, code):
        self.__thread.exit(code)
        


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    #app.setStyleSheet('''QPushButton {
                            #border: 2px solid #8f8f91;
                            #border-radius: 6px;
                            #background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                            #stop: 0 #f6f7fa, stop: 1 #dadbde);
                            #padding: 1px 4px;
                      #}''')

    k = Qt.QStyleFactory.keys()
    base = os.path.expanduser('~/data/xsocs/id01_data/psic_nano_20150314_fast_00007')
    #spec_file = os.path.join(base, 'psic_nano_20150314_fast_00007.spec')
    spec_file='/users/naudet/workspace/dau/id01/tests/gui/test_spec2.spec'
    mw = MergeWidget(spec_file=spec_file,
                     img_dir='/users/naudet/data/xsocs/id01_data/psic_nano_20150314_fast_00007/004_200',
                     output_dir='/users/naudet/workspace/dau/id01/tests/gui/out',
                     spec_version=0)
    #mw.setStyle(Qt.QStyleFactory.create(i))
    mw.show()

    app.exec_()


#label.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Raised)
#sp = total_line_edit.sizePolicy()
#sp.setHorizontalPolicy(Qt.QSizePolicy.Minimum)
#total_line_edit.setSizePolicy(sp)
