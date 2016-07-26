from functools import partial
from collections import namedtuple

from kmap.process.qspace import RecipSpaceConverter

from silx.gui import qt as Qt

_MAX_BEAM_ENERGY_EV = 10**6

_MU_LOWER = u'\u03BC'
_PHI_LOWER = u'\u03C6'
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
        imgbin_h_edit = _AdjustedLineEdit(5)
        imgbin_h_edit.setValidator(Qt.QIntValidator(imgbin_h_edit))
        imgbin_h_edit.setAlignment(Qt.Qt.AlignRight)
        imgbin_h_edit.setText(str(_DEFAULT_IMG_BIN[0]))
        h_layout.addWidget(imgbin_h_edit, alignment=Qt.Qt.AlignLeft)
        h_layout.addWidget(Qt.QLabel(' x '))
        imgbin_v_edit = _AdjustedLineEdit(5)
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
        qsize_x_edit = _AdjustedLineEdit(5)
        qsize_x_edit.setValidator(Qt.QDoubleValidator(qsize_x_edit))
        qsize_x_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_x_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_y_edit = _AdjustedLineEdit(5)
        qsize_y_edit.setValidator(Qt.QDoubleValidator(qsize_y_edit))
        qsize_y_edit.setAlignment(Qt.Qt.AlignRight)
        h_layout.addWidget(qsize_y_edit)
        h_layout.addWidget(Qt.QLabel(' x '))
        qsize_z_edit = _AdjustedLineEdit(5)
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


class AcqParamsWidget(Qt.QWidget):

    def __init__(self, read_only=False, **kwargs):
        super(AcqParamsWidget, self).__init__(**kwargs)
        layout = Qt.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.__beam_energy = None
        self.__dir_beam_h = None
        self.__dir_beam_v = None
        self.__chpdeg_h = None
        self.__chpdeg_v = None
        self.__pixelsize_h = None
        self.__pixelsize_v = None
        self.__detector_orient = None

        class DblValidator(Qt.QDoubleValidator):
            def validate(self, text, pos):
                if len(text) == 0:
                    return Qt.QValidator.Acceptable, text, pos
                return super(DblValidator, self).validate(text, pos)

        def dblLineEditWidget(width):
            wid = _AdjustedLineEdit(width,
                                    validator_cls=DblValidator,
                                    read_only=read_only,
                                    reset_on_empty=True,
                                    highlight_change=True,
                                    field_type=float)
            wid.setReadOnly(read_only)
            return wid

        # ===========
        # beam energy
        # ===========
        row = 0
        h_layout = Qt.QHBoxLayout()
        beam_nrg_edit = dblLineEditWidget(8)
        h_layout.addWidget(beam_nrg_edit)
        h_layout.addWidget(Qt.QLabel('<b>eV</b>'))

        layout.addWidget(Qt.QLabel('Beam energy :'), row, 0)
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 2)

        # ===========
        # direct beam
        # ===========

        row += 1
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        dir_beam_h_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel('h='))
        h_layout.addWidget(dir_beam_h_edit)
        dir_beam_v_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel(' v='))
        h_layout.addWidget(dir_beam_v_edit)
        h_layout.addWidget(Qt.QLabel('<b>px</b>'))
        layout.addWidget(Qt.QLabel('Direct beam :'), row, 0)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # chan per degree
        # ===========

        row += 1
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        chpdeg_h_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel('h='))
        h_layout.addWidget(chpdeg_h_edit)
        chpdeg_v_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel(' v='))
        h_layout.addWidget(chpdeg_v_edit)
        h_layout.addWidget(Qt.QLabel('<b>px</b>'))
        layout.addWidget(Qt.QLabel('Chan. per deg. :'), row, 0)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # pixelsize
        # ===========

        row += 1
        h_layout = Qt.QHBoxLayout()
        layout.addLayout(h_layout, row, 1,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)
        pixelsize_h_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel('h='))
        h_layout.addWidget(pixelsize_h_edit)
        pixelsize_v_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel(' v='))
        h_layout.addWidget(pixelsize_v_edit)
        h_layout.addWidget(Qt.QLabel(u'<b>\u03BCm</b>'))
        layout.addWidget(Qt.QLabel('Pixel size. :'), row, 0)

        # ===

        row += 1
        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        layout.addWidget(h_line, row, 0, 1, 3)

        # ===========
        # detector orientation
        # ===========

        row += 1
        layout.addWidget(Qt.QLabel('Det. orientation :'), row, 0)
        h_layout = Qt.QHBoxLayout()
        det_phi_rb = Qt.QRadioButton(u'Width is {0}.'.format(_PHI_LOWER))
        h_layout.addWidget(det_phi_rb)
        det_mu_rb = Qt.QRadioButton(u'Width is {0}.'.format(_MU_LOWER))
        h_layout.addWidget(det_mu_rb)
        layout.addLayout(h_layout, row, 1)
        det_phi_rb.setEnabled(not read_only)
        det_mu_rb.setEnabled(not read_only)

        # ===========
        # size constraints
        # ===========
        self.setSizePolicy(Qt.QSizePolicy(Qt.QSizePolicy.Fixed,
                                          Qt.QSizePolicy.Fixed))

        # named tuple with references to all the important widgets
        self.__beam_nrg_edit = beam_nrg_edit
        self.__dir_beam_h_edit = dir_beam_h_edit
        self.__dir_beam_v_edit = dir_beam_v_edit
        self.__chpdeg_h_edit = chpdeg_h_edit
        self.__chpdeg_v_edit = chpdeg_v_edit
        self.__pixelsize_h_edit = pixelsize_h_edit
        self.__pixelsize_v_edit = pixelsize_v_edit
        self.__det_phi_rb = det_phi_rb
        self.__det_mu_rb = det_mu_rb

    @property
    def beam_energy(self):
        text = self.__beam_nrg_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @beam_energy.setter
    def beam_energy(self, beam_energy):
        self.__beam_nrg_edit.setText(str(beam_energy))
        self.__beam_energy = beam_energy

    @property
    def direct_beam_h(self):
        text = self.__dir_beam_h_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @direct_beam_h.setter
    def direct_beam_h(self, direct_beam_h):
        self.__dir_beam_h_edit.setText(str(direct_beam_h))
        self.__dir_beam_h = direct_beam_h

    @property
    def direct_beam_v(self):
        text = self.__dir_beam_v_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @direct_beam_v.setter
    def direct_beam_v(self, direct_beam_v):
        self.__dir_beam_v_edit.setText(str(direct_beam_v))
        self.__dir_beam_v = direct_beam_v

    @property
    def chperdeg_h(self):
        text = self.__chpdeg_h_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @chperdeg_h.setter
    def chperdeg_h(self, chperdeg_h):
        self.__chpdeg_h_edit.setText(str(chperdeg_h))
        self.__chpdeg_h = chperdeg_h

    @property
    def chperdeg_v(self):
        text = self.__chpdeg_v_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @chperdeg_v.setter
    def chperdeg_v(self, chperdeg_v):
        self.__chpdeg_v_edit.setText(str(chperdeg_v))
        self.__chpdeg_v = chperdeg_v

    @property
    def pixelsize_h(self):
        return 21
        text = self.__pixelsize_h_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @pixelsize_h.setter
    def pixelsize_h(self, pixelsize_h):
        self.__pixelsize_h_edit.setText(str(pixelsize_h))
        self.__pixelsize_h = pixelsize_h

    @property
    def pixelsize_v(self):
        text = self.__pixelsize_v_edit.text()
        if len(text) == 0:
            return None
        return float(text)

    @pixelsize_v.setter
    def pixelsize_v(self, pixelsize_v):
        self.__pixelsize_v_edit.setText(str(pixelsize_v))
        self.__pixelsize_v = pixelsize_v

    def __getDetectorOrient(self):
        if self.__det_phi_rb.isChecked():
            return 'phi'
        elif self.__det_mu_rb.isChecked():
            return 'mu'
        return None

    def __setDetectorOrient(self, detector_orient):
        if detector_orient == 'phi':
            self.__det_phi_rb.setChecked(True)
        elif detector_orient == 'mu':
            self.__det_mu_rb.setChecked(True)
        else:
            self.__det_phi_rb.setChecked(False)
            self.__det_mu_rb.setChecked(False)
            return
            # raise ValueError('Unknown detector orientation : {0}.'
            #                  ''.format(detector_orient))
        self.__detector_orient = detector_orient

    detector_orient = property(__getDetectorOrient, __setDetectorOrient)


class RecipSpaceWidget(Qt.QWidget):

    __sigConvertDone = Qt.Signal()

    def __init__(self,
                 data_h5f=None,
                 output_f=None,
                 qspace_size=None,
                 image_binning=None,
                 **kwargs):
        super(Qt.QWidget, self).__init__(**kwargs)
        Qt.QGridLayout(self)

        # ATTENTION : this is done to allow the stretch
        # of the QTableWidget containing the scans info
        self.layout().setColumnStretch(1, 1)

        # ################
        # input QGroupBox
        # ################

        input_gbx = Qt.QGroupBox("Input")
        layout = Qt.QHBoxLayout(input_gbx)
        self.layout().addWidget(input_gbx,
                                0, 0,
                                1, 2)

        # data HDF5 file input
        lab = Qt.QLabel('HDF5 file :')
        h5_file_edit = Qt.QLineEdit()
        fm = h5_file_edit.fontMetrics()
        h5_file_edit.setMinimumWidth(fm.width(' ' * 100))
        h5_file_bn = _AdjustedPushButton('...')
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
        scans_gbx = Qt.QGroupBox("Scans")
        self.layout().addWidget(scans_gbx, 1, 1, 2, 1)
        self.layout().setRowStretch(2, 1000)

        grp_layout = Qt.QVBoxLayout(scans_gbx)
        info_layout = Qt.QGridLayout()
        grp_layout.addLayout(info_layout)

        fm = self.fontMetrics()

        label = Qt.QLabel(u'# points :')
        n_img_label = _AdjustedLabel(5)
        info_layout.addWidget(label, 0, 0)
        info_layout.addWidget(n_img_label, 0, 1)

        label = Qt.QLabel(u'# {0} :'.format(_ETA_LOWER))
        n_angles_label = _AdjustedLabel(5)
        info_layout.addWidget(label, 1, 0)
        info_layout.addWidget(n_angles_label, 1, 1)
        info_layout.setColumnStretch(2, 1)

        scans_table = Qt.QTableWidget(0, 2)
        scans_table.verticalHeader().hide()
        grp_layout.addWidget(scans_table)

        # ################
        # Acq. parameters
        # ################
        params_gbx = Qt.QGroupBox("Acq. Parameters")
        grp_layout = Qt.QVBoxLayout(params_gbx)
        self.layout().addWidget(params_gbx,
                                1, 0, alignment=Qt.Qt.AlignTop)

        def_acqparam_bn = Qt.QCheckBox('Use input file values')
        def_acqparam_bn.setCheckState(Qt.Qt.Checked)
        grp_layout.addWidget(def_acqparam_bn)

        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        grp_layout.addWidget(h_line)

        param_stacked = Qt.QStackedWidget()
        grp_layout.addWidget(param_stacked)

        acq_params_ro = AcqParamsWidget(read_only=True)
        param_stacked.addWidget(acq_params_ro)

        acq_params_rw = AcqParamsWidget()
        param_stacked.addWidget(acq_params_rw)

        grp_layout.addStretch(1)

        # ################
        # conversion params
        # ################

        conv_gbx = Qt.QGroupBox("Conversion parameters")
        grp_layout = Qt.QVBoxLayout(conv_gbx)
        self.layout().addWidget(conv_gbx,
                                2, 0, alignment=Qt.Qt.AlignTop)

        conv_params_wid = ConversionParamsWidget()
        grp_layout.addWidget(conv_params_wid)

        # ################
        # output
        # ################

        output_gbx = Qt.QGroupBox("Output")
        layout = Qt.QHBoxLayout(output_gbx)
        self.layout().addWidget(output_gbx,
                                3, 0,
                                1, 2)
        lab = Qt.QLabel('File :')
        output_file_edit = Qt.QLineEdit()
        fm = output_file_edit.fontMetrics()
        output_file_edit.setMinimumWidth(fm.width(' ' * 100))
        output_file_bn = _AdjustedPushButton('...')
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
        self.layout().addLayout(h_layout,
                                4, 0,
                                1, 2,
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
                                  'params_gbx',
                                  'conv_gbx',
                                  'output_gbx',
                                  'n_img_label',
                                  'n_angles_label',
                                  'scans_table',
                                  'def_acqparam_bn',
                                  'param_stacked',
                                  'acq_params_ro',
                                  'acq_params_rw',
                                  'conv_params_wid',
                                  'output_file_edit',
                                  'output_file_bn',
                                  'convert_bn'])
        self.__widgets = SelfWidgets(h5_file_edit=h5_file_edit,
                                     h5_file_bn=h5_file_bn,
                                     scans_gbx=scans_gbx,
                                     params_gbx=params_gbx,
                                     conv_gbx=conv_gbx,
                                     output_gbx=output_gbx,
                                     n_img_label=n_img_label,
                                     n_angles_label=n_angles_label,
                                     scans_table=scans_table,
                                     def_acqparam_bn=def_acqparam_bn,
                                     param_stacked=param_stacked,
                                     acq_params_ro=acq_params_ro,
                                     acq_params_rw=acq_params_rw,
                                     conv_params_wid=conv_params_wid,
                                     output_file_edit=output_file_edit,
                                     output_file_bn=output_file_bn,
                                     convert_bn=convert_bn)

        cancel_bn.clicked.connect(self.close)
        h5_file_bn.clicked.connect(self.__pickInputFile)
        def_acqparam_bn.stateChanged.connect(self.__acqParamChkBnStateChanged)
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

    def __convertBnClicked(self, *args, **kwargs):
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

        if not widgets.def_acqparam_bn.isChecked():
            beam_energy = widgets.acq_params_rw.beam_energy
            direct_beam_h = widgets.acq_params_rw.direct_beam_h
            direct_beam_v = widgets.acq_params_rw.direct_beam_v
            chperdeg_h = widgets.acq_params_rw.chperdeg_h
            chperdeg_v = widgets.acq_params_rw.chperdeg_v
            pixelsize_h = widgets.acq_params_rw.pixelsize_h
            pixelsize_v = widgets.acq_params_rw.pixelsize_v
            detector_orient = widgets.acq_params_rw.detector_orient
            kwargs = dict(beam_energy=beam_energy,
                          chan_per_deg=[chperdeg_h, chperdeg_v],
                          center_chan=[direct_beam_h, direct_beam_v],
                          detector_orient=detector_orient,
                          pixelsize=[pixelsize_h, pixelsize_v])
        else:
            kwargs = {}

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
        _ConversionProcessDialog(converter, parent=self, **kwargs).exec_()

    def __acqParamChkBnStateChanged(self, state):
        """
        Sets the current acquisition parameters widget shown
        in the QStackedWidget
        """
        widgets = self.__widgets
        param_stacked = widgets.param_stacked
        if state == Qt.Qt.Checked:
            current_widget = widgets.acq_params_ro
        else:
            current_widget = widgets.acq_params_rw
        param_stacked.setCurrentWidget(current_widget)

    def __resetState(self):
        widgets = self.__widgets

        widgets.scans_table.clear()
        widgets.scans_table.setHorizontalHeaderLabels(['scan', 'eta'])

        widgets.n_img_label.setText('')
        widgets.n_angles_label.setText('')
        widgets.def_acqparam_bn.setChecked(True)

        self.__groupsSetEnabled(False)

        widgets.output_file_edit.setText('')

    def __pickOutputFile(self, *args, **kwargs):
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

    def __pickInputFile(self, *args, **kwargs):
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
            print 'EX', ex
            raise ex

        self.__converter = converter

        self.__fillScansInfos()
        self.__fillAcqParamWidgets()
        self.__groupsSetEnabled(True)

    def __fillAcqParamWidgets(self):
        """
        Fills both AcqParamWidgets (read only and editable) with
        info found in the input file
        """
        converter = self.__converter
        if converter is None:
            return
        scans = converter.scans
        params = converter.scan_params(scans[0])

        widgets = self.__widgets
        widgets.def_acqparam_bn.setChecked(True)

        ro_wid = widgets.acq_params_ro
        rw_wid = widgets.acq_params_rw

        direct_beam = params['center_chan']
        chperdeg = params['chan_per_deg']
        pixelsize = params['pixelsize']

        for wid in [ro_wid, rw_wid]:
            wid.beam_energy = params['beam_energy']
            wid.detector_orient = params['detector_orient']
            wid.direct_beam_h = direct_beam[0]
            wid.direct_beam_v = direct_beam[1]
            wid.chperdeg_h = chperdeg[0]
            wid.chperdeg_v = chperdeg[1]
            wid.pixelsize_h = pixelsize[0]
            wid.pixelsize_v = pixelsize[1]

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

        params = converter.scan_params(scans[0])
        widgets.n_img_label.setText(str(params['n_images']))
        widgets.n_angles_label.setText(str(len(scans)))

    def __groupsSetEnabled(self, enable=True):
        widgets = self.__widgets
        widgets.scans_gbx.setEnabled(enable)
        widgets.params_gbx.setEnabled(enable)
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

        bn_box = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Abort)
        bn_box.button(Qt.QDialogButtonBox.Ok).setText('Done')
        bn_box.button(Qt.QDialogButtonBox.Ok).setEnabled(False)
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
                          # pos_indices=range(10),
                          **kwargs)
        self.__qtimer.start(1000)

    def __onAbort(self):
        self.__status_lab.setText('<font color="orange">Cancelling...</font>')
        self.__converter.abort(wait=False)
        self.__aborted = True

    def __onProgress(self):
        progress = self.__converter.progress()
        self.__progress_bar.setValue(progress)

    def __convertDone(self):
        self.__qtimer.stop()
        self.__qtimer = None
        self.__onProgress()
        if self.__aborted:
            self.__status_lab.setText('<font color="red">Conversion '
                                      'cancelled.</font>')
        else:
            self.__status_lab.setText('<font color="green">Conversion '
                                      'done.</font>')
        self.__bn_box.button(Qt.QDialogButtonBox.Ok).setText('Done')
        self.__bn_box.button(Qt.QDialogButtonBox.Ok).setEnabled(True)
        self.__bn_box.button(Qt.QDialogButtonBox.Abort).setEnabled(False)


def before_after(f):
    def decorator(*args, **kwargs):
        print('before', f.func_name)
        f(*args, **kwargs)
        print('after', f.func_name)
    return decorator


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
    def __init__(self,
                 width,
                 padding=None,
                 alignment=Qt.Qt.AlignRight,
                 validator_cls=None,
                 field_type=None,
                 read_only=False,
                 reset_on_empty=False,
                 highlight_change=False,
                 **kwargs):
        super(_AdjustedLineEdit, self).__init__(**kwargs)

        self.__defaultText = self.text()
        self.__highlightChange = highlight_change
        self.__resetOnEmpty = reset_on_empty
        self.__fieldType = field_type

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        text = '0' * width
        width = fm.width(text) + padding
        self.setMaximumWidth(width)

        self.setAlignment(alignment)
        self.setReadOnly(read_only)

        if validator_cls is not None:
            self.setValidator(validator_cls(self))

        self.setStyleSheet('_AdjustedLineEdit[readOnly="true"][enabled="true"]'
                           '{ background-color: lightGray; }')

        if not self.isReadOnly():
            self.textChanged.connect(self.__onTextChanged)
            self.textEdited.connect(self.__updateField)
            self.editingFinished.connect(partial(self.__updateField,
                                                 finished=True))
            self.returnPressed.connect(partial(self.__updateField,
                                               finished=True))

    def __updateField(self, text=None, finished=False):
        if text is None:
            value = self.text()
        else:
            value = text

        if len(value) > 0:
            pass
        elif finished and self.__resetOnEmpty:
            self.resetDefault(block=True)
            value = self.text()
        else:
            pass

        if len(value) == 0:
            if len(self.__defaultText) == 0:
                same_txt = True
            else:
                same_txt = False
        elif len(self.__defaultText) == 0:
            same_txt = False
        else:
            if self.__fieldType is not None:
                try:
                    value = self.__fieldType(value)
                    default_value = self.__fieldType(self.__defaultText)
                except:
                    same_txt = False
            else:
                default_value = self.__defaultText
            if value == default_value:
                same_txt = True
            else:
                same_txt = False

        if self.__highlightChange and not same_txt:
            self.setStyleSheet('_AdjustedLineEdit {'
                               'background-color: lightblue;}')
        else:
            self.setStyleSheet('')

    def __onEditingFinished(self):
        self.__updateField(finished=True)

    def __onTextEdited(self, text):
        self.__updateField(text)

    def __onTextChanged(self, text):
        if self.isModified() is False:
            self.__defaultText = text
            self.__updateField()

    def resetDefault(self, block=False):
        if block:
            self.textChanged.disconnect(self.__onTextChanged)
        self.setText(self.__defaultText)
        if block:
            self.textChanged.connect(self.__onTextChanged)

    defaultText = property(lambda self: self.__defaultText)

    def event(self, ev):
        # this has to be done so that the stylesheet is reapplied when the
        # "enabled" property changes
        # https://wiki.qt.io/Dynamic_Properties_and_Stylesheets
        if ev.type() == Qt.QEvent.EnabledChange:
            self.style().unpolish(self)
            self.style().polish(self)
        return super(_AdjustedLineEdit, self).event(ev)


class _AdjustedLabel(Qt.QLabel):
    """
    """
    def __init__(self,
                 width,
                 padding=None,
                 alignment=Qt.Qt.AlignRight,
                 **kwargs):
        super(_AdjustedLabel, self).__init__(**kwargs)

        self.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        text = '0' * width
        width = fm.width(text) + padding
        self.setMinimumWidth(width)

        self.setAlignment(alignment)


if __name__ == '__main__':
    pass
