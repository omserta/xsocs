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
from ..widgets.RangeSlider import RangeSlider
from ..widgets.Input import StyledLineEdit, FixedSizeLabel


class RoiAxisWidget(Qt.QWidget):
    """
    Widget with a double slider and two line edit
    displaying the slider range.
    """

    sigSliderMoved = Qt.Signal(object)
    """ Signal triggered when the slider is moved. Equivalent to connecting
    directly to the sliders sigSliderMoved signal.
    """

    def slider(self):
        """
        The RangeSlider instance of this widget.
        :return:
        """
        return self.__slider

    def __init__(self, label=None, **kwargs):
        """

        :param label: text displayed above the slider.
        :param kwargs:
        """
        super(RoiAxisWidget, self).__init__(**kwargs)

        layout = Qt.QGridLayout(self)
        qLabel = FixedSizeLabel(nChar=1)
        qLabel.setFrameStyle(Qt.QFrame.NoFrame | Qt.QFrame.Plain)
        qLabel.setText(label)
        slider = self.__slider = RangeSlider()
        leftEdit = self.__leftEdit = StyledLineEdit(nChar=7)
        rightEdit = self.__rightEdit = StyledLineEdit(nChar=7)
        leftEdit.setReadOnly(True)
        rightEdit.setReadOnly(True)

        layout.addWidget(qLabel, 0, 0)
        layout.addWidget(slider, 0, 1, 1, 2)
        layout.addWidget(leftEdit, 1, 1)
        layout.addWidget(rightEdit, 1, 2)

        layout.setColumnStretch(3, 1)

        slider.sigSliderMoved.connect(self.__sliderMoved)
        slider.sigSliderMoved.connect(self.sigSliderMoved)

    def __sliderMoved(self, event):
        """
        Slot triggered when one of the slider is moved. Updates the
        line edits.
        :param event:
        :return:
        """
        self.__leftEdit.setText('{0:6g}'.format(event.left))
        self.__rightEdit.setText('{0:6g}'.format(event.right))


if __name__ == '__main__':
    pass
