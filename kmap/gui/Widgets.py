from functools import partial
from silx.gui import qt as Qt

_MU_LOWER = u'\u03BC'
_PHI_LOWER = u'\u03C6'
_ETA_LOWER = u'\u03B7'


class AcqParamsWidget(Qt.QWidget):

    def __init__(self,
                 read_only=False,
                 highlight_change=True,
                 **kwargs):
        super(AcqParamsWidget, self).__init__(**kwargs)
        layout = Qt.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.__read_only = read_only

        self.__beam_energy = None
        self.__dir_beam_h = None
        self.__dir_beam_v = None
        self.__chpdeg_h = None
        self.__chpdeg_v = None

        class DblValidator(Qt.QDoubleValidator):
            def validate(self, text, pos):
                if len(text) == 0:
                    return Qt.QValidator.Acceptable, text, pos
                return super(DblValidator, self).validate(text, pos)

        def dblLineEditWidget(width):
            wid = AdjustedLineEdit(width,
                                   validator_cls=DblValidator,
                                   read_only=read_only,
                                   reset_on_empty=True,
                                   highlight_change=highlight_change,
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
        h_layout.addWidget(Qt.QLabel('v='))
        h_layout.addWidget(dir_beam_h_edit)
        dir_beam_v_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel(' h='))
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
        h_layout.addWidget(Qt.QLabel('v='))
        h_layout.addWidget(chpdeg_h_edit)
        chpdeg_v_edit = dblLineEditWidget(6)
        h_layout.addWidget(Qt.QLabel(' h='))
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

    def clear(self):
        self.__beam_nrg_edit.clear()
        self.__dir_beam_h_edit.clear()
        self.__dir_beam_v_edit.clear()
        self.__chpdeg_h_edit.clear()
        self.__chpdeg_v_edit.clear()

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


class AdjustedPushButton(Qt.QPushButton):
    """
    It seems that by default QPushButtons minimum width is 75.
    This is a workaround.
    For _AdjustedPushButton to work text has to be set at creation time.
    """
    def __init__(self, text, parent=None, padding=None):
        super(AdjustedPushButton, self).__init__(text, parent)

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        width = fm.width(self.text()) + padding
        self.setMaximumWidth(width)


class AdjustedLineEdit(Qt.QLineEdit):
    """
    """
    def __init__(self,
                 width=None,
                 parent=None,
                 padding=None,
                 alignment=Qt.Qt.AlignRight,
                 validator_cls=None,
                 field_type=None,
                 read_only=False,
                 reset_on_empty=False,
                 highlight_change=False):
        super(AdjustedLineEdit, self).__init__(parent)

        self.__defaultText = self.text()
        self.__highlightChange = highlight_change
        self.__resetOnEmpty = reset_on_empty
        self.__fieldType = field_type

        if width is not None:
            fm = self.fontMetrics()

            if padding is None:
                padding = 2 * fm.width('0')

            text = '0' * width
            width = fm.width(text) + padding
            self.setMaximumWidth(width)
            self.setMinimumWidth(width)

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
                    # TODO : filter specific exception
                    default_value = None
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
        return super(AdjustedLineEdit, self).event(ev)


class AdjustedLabel(Qt.QLabel):
    """
    """
    def __init__(self,
                 width,
                 padding=None,
                 alignment=Qt.Qt.AlignRight,
                 **kwargs):
        super(AdjustedLabel, self).__init__(**kwargs)

        self.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)

        fm = self.fontMetrics()

        if padding is None:
            padding = 2 * fm.width('0')

        text = '0' * width
        width = fm.width(text) + padding
        self.setMinimumWidth(width)

        self.setAlignment(alignment)
