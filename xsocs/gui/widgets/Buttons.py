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


class FixedSizePushButon(Qt.QPushButton):
    """
    It seems that by default QPushButtons minimum width is 75.
    This is a workaround.
    """

    _emptyWidth = 4
    _padding = 1

    def __init__(self, *args, **kwargs):
        super(FixedSizePushButon, self).__init__(*args, **kwargs)
        self.__nChar = None
        self._updateSize()

    def setText(self, text):
        """
        Reimplemented from QPushButton::setText.
        Calls _updateStyleSheet
        :param text:
        :return:
        """
        super(FixedSizePushButon, self).setText(text)
        self._updateSize()

    def _updateSize(self):
        """
        Sets the widget's size
        :return:
        """

        style = Qt.QApplication.style()
        border = 2 * (self._padding +
                      style.pixelMetric(Qt.QStyle.PM_ButtonMargin,
                                        widget=self))

        fm = self.fontMetrics()
        width = fm.width(self.text()) + border
        self.setFixedWidth(width)


if __name__ == '__main__':
    pass
