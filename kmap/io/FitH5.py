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

import numpy as np

from .XsocsH5Base import XsocsH5Base


class FitH5(XsocsH5Base):
    scan_positions_path = 'scan_positions'
    q_fit_path = 'q{0}fit'
    status_path = 'success'

    @property
    def scan_positions(self):
        with self:
            return self.sample_x, self.sample_y

    sample_x = property(lambda self:
                        self._get_array_data(FitH5.scan_positions_path + '/x'))

    sample_y = property(lambda self:
                        self._get_array_data(FitH5.scan_positions_path + '/y'))

    x_fit = property(lambda self: self.__get_fit('x'))

    y_fit = property(lambda self: self.__get_fit('y'))

    z_fit = property(lambda self: self.__get_fit('z'))

    x_height = property(lambda self: self.__get_data('x', 'height'))

    y_height = property(lambda self: self.__get_data('y', 'height'))

    z_height = property(lambda self: self.__get_data('z', 'height'))

    x_center = property(lambda self: self.__get_data('x', 'center'))

    y_center = property(lambda self: self.__get_data('y', 'center'))

    z_center = property(lambda self: self.__get_data('z', 'center'))

    x_width = property(lambda self: self.__get_data('x', 'width'))

    y_width = property(lambda self: self.__get_data('y', 'width'))

    z_width = property(lambda self: self.__get_data('z', 'width'))

    status = property(lambda self: self._get_array_data(FitH5.status_path))

    def __get_fit(self, axis):
        with self:
            height = self.__get_data(axis, 'height')
            center = self.__get_data(axis, 'center')
            width = self.__get_data(axis, 'width')
        return height, center, width

    def __get_data(self, axis, data):
        data_path = FitH5.q_fit_path.format(axis) + '/{0}'.format(data)
        return self._get_array_data(data_path)

    def export_txt(self, filename):
        with self:
            with open(filename, 'w+') as res_f:
                res_f.write('X Y '
                            'height_x center_x width_x '
                            'height_y center_y width_y '
                            'height_z center_z width_z '
                            '|q| status\n')
                x_height, x_center, x_width = self.x_fit
                y_height, y_center, y_width = self.y_fit
                z_height, z_center, z_width = self.z_fit
                q = np.sqrt(x_center ** 2 +
                            y_center ** 2 +
                            z_center ** 2)
                status = self.status
                x, y = self.scan_positions

                for i, s in enumerate(status):
                    r = [x[i], y[i],
                         x_height[i], x_center[i], x_width[i],
                         y_height[i], y_center[i], y_width[i],
                         z_height[i], z_center[i], z_width[i],
                         q[i], s]
                    res_str = '{0}\n'.format(' '.join(str(e) for e in r))
                    res_f.write(res_str)


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
