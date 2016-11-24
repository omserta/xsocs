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

import weakref
from contextlib import contextmanager

import numpy as _np

from .XsocsH5Base import XsocsH5Base


class FitH5(XsocsH5Base):
    scan_positions_path = 'scan_positions'
    q_fit_path = 'q{0}fit'
    status_path = 'success'

    def scan_positions(self):
        path_tpl = FitH5.scan_positions_path + '/{0}'
        x = self._get_array_data(path_tpl.format('x'))
        y = self._get_array_data(path_tpl.format('y'))
        return x, y

    def x_fit(self):
        return self.__get_fit('x')

    def y_fit(self):
        return self.__get_fit('y')

    def z_fit(self):
        return self.__get_fit('z')

    def __get_fit(self, axis):
        height = self.__get_data(axis, 'height')
        center = self.__get_data(axis, 'center')
        width = self.__get_data(axis, 'width')
        return height, center, width

    def x_height(self):
        return self.__get_data('x', 'height')

    def x_center(self):
        return self.__get_data('x', 'center')

    def x_width(self):
        return self.__get_data('x', 'width')
        
    def y_height(self):
        return self.__get_data('y', 'height')

    def y_center(self):
        return self.__get_data('y', 'center')

    def y_width(self):
        return self.__get_data('y', 'width')
        
    def z_height(self):
        return self.__get_data('z', 'height')

    def z_center(self):
        return self.__get_data('z', 'center')

    def z_width(self):
        return self.__get_data('z', 'width')

    def __get_data(self, axis, data):
        data_path = FitH5.q_fit_path.format(axis) + '/{0}'.format(data)
        return self._get_array_data(data_path)

    def set_status(self, status):
        return self._get_array_data(FitH5.status_path)


class FitH5Writer(FitH5):

    def __init__(self, h5_f, mode='a', **kwargs):
        super(FitH5Writer, self).__init__(h5_f, mode=mode, **kwargs)

    def set_scan_positions(self, x, y):
        path_tpl = FitH5.scan_positions_path + '/{0}'
        self._set_array_data(path_tpl.format('x'), x)
        self._set_array_data(path_tpl.format('y'), y)

    def set_x_fit(self, height, center, width):
        self.__set_fit('x', height, center, width)

    def set_y_fit(self, height, center, width):
        self.__set_fit('y', height, center, width)

    def set_z_fit(self, height, center, width):
        self.__set_fit('z', height, center, width)

    def __set_fit(self, axis, height, center, width):
        path_tpl = FitH5.q_fit_path.format(axis) + '/{0}'
        self._set_array_data(path_tpl.format('height'), height)
        self._set_array_data(path_tpl.format('center'), center)
        self._set_array_data(path_tpl.format('width'), width)

    def set_status(self, status):
        self._set_array_data(FitH5.status_path, status)
