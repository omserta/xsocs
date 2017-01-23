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

from .Input import StyledLineEdit

_MU_LOWER = u'\u03BC'
_PHI_LOWER = u'\u03C6'
_ETA_LOWER = u'\u03B7'


class AcqParamsWidget(Qt.QWidget):

    def __init__(self,
                 read_only=False,
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
            wid = StyledLineEdit(nChar=width,
                                 readOnly=read_only)
            wid.setValidator(DblValidator())

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

if __name__ == '__main__':
    pass