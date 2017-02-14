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

import sys

from silx.gui import qt as Qt
print('Using Qt {0}'.format(Qt.qVersion()))

from .XsocsGui import XsocsGui
from .process.MergeWidget import MergeWidget
from .process.QSpaceWidget import QSpaceWidget


def merge_window(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    mw = MergeWidget(*args, **kwargs)
    mw.show()
    app.exec_()


def conversion_window(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    mw = QSpaceWidget(*args, **kwargs)
    mw.show()
    app.exec_()


def xsocs_main(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    mw = XsocsGui(*args, **kwargs)
    mw.show()
    app.exec_()
