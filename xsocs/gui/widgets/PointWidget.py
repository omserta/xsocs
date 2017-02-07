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
from .Input import FixedSizeLabel, StyledLineEdit


class PointWidget(Qt.QFrame):
    """
    Widget displaying 2 values (x, y).
    """
    def __init__(self, **kwargs):
        super(PointWidget, self).__init__(**kwargs)

        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.__xEdit = xEdit = StyledLineEdit(nChar=6)
        xEdit.setReadOnly(True)
        self.__yEdit = yEdit = StyledLineEdit(nChar=6)
        yEdit.setReadOnly(True)

        xLabel = 'x'
        yLabel = 'y'

        layout.addWidget(FixedSizeLabel(xLabel, nChar=len(xLabel)))
        layout.addWidget(xEdit)
        layout.addWidget(FixedSizeLabel(yLabel, nChar=len(yLabel)))
        layout.addWidget(yEdit)

    def setPoint(self, x, y):
        """
        Sets the values to display.
        :param x:
        :param y:
        :return:
        """
        self.__xEdit.setText('{0:6g}'.format(x))
        self.__yEdit.setText('{0:6g}'.format(y))
