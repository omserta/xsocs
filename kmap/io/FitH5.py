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


# class GaussianFitH5(FitH5):

class FitH5(XsocsH5Base):
    title_path = '{entry}/title'
    start_time_path = '{entry}/start_time'
    end_time_path = '{entry}/end_time'
    date_path = '{entry}/{process}/date'
    axis_path = '{entry}/{process}/axis'
    configuration_path = '{entry}/{process}/configuration'
    results_path = '{entry}/{process}/results'
    qx_results_path = '{entry}/{process}/results/qx/{0}'
    qy_results_path = '{entry}/{process}/results/qy/{0}'
    qz_results_path = '{entry}/{process}/results/qz/{0}'

    def __init__(self, h5_f, mode='r'):
        super(XsocsH5Base, self).__init__(h5_f, mode=mode)

        self.__entries = None

    def title(self, entry):
        with self._get_file() as h5_file:
            path = entry + '/title'
            return h5_file[path][()]

    def _update_entries(self):
        with self._get_file() as h5_file:
            # TODO : this isnt pretty but for some reason the attrs.get() fails
            # when there is no attribute NX_class (should return the default
            # None)
            self.__entries = sorted([key for key in h5_file
                                     if ('NX_class' in h5_file[key].attrs and
                                         h5_file[key].attrs['NX_class'] == 'NXentry')])  # noqa

    def entries(self):
        if self.__entries is None:
            self._update_entries()
        return self.__entries[:]

    def processes(self, entry):
        with self._get_file() as h5_file:
            entry_grp = h5_file[entry]
            processes = sorted([key for key in entry_grp
                                if ('NX_class' in entry_grp[key].attrs and
                                    h5_file[key].attrs[
                                        'NX_class'] == 'NXprocess')])
        return processes

    def results(self, entry, process):
        with self._get_file() as h5_file:
            results_path = FitH5.results_path.format(entry=entry,
                                                     process=process)
            result_grp = h5_file[results_path]
            results = sorted([key for key in result_grp])
            return results

    def result(self, entry, process, result):
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=result)
        return self._get_array_data(result_path)


class FitH5Writer(FitH5):

    def create_entry(self, entry):
        with self._get_file() as h5_file:
            # TODO : check if it already exists
            entry_grp = h5_file.require_group(entry)
            entry_grp.attrs['NX_class'] = np.string_('NXentry')
            self._update_entries()

    def create_process(self, entry, process):
        # TODO : check that there isn't already an existing process
        with self._get_file() as h5_file:
            if entry not in h5_file:
                self.create_entry(entry)
            entry_grp = h5_file[entry]

            # TODO : check if it exists
            process_grp = entry_grp.require_group[process]
            process_grp.attrs['NX_class'] = np.string_('NXprocess')
            results_grp = process_grp.require_group('results')
            results_grp.attrs['NX_class'] = np.string_('NXcollection')

    def set_title(self, entry, title):
        self._set_scalar_data(FitH5.title_path.format(entry), title)

    def set_result(self, entry, process, name, data):
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=name)
        self._set_array_data(result_path, data)

    #
    # def export_txt(self, filename):
    #     with self:
    #         with open(filename, 'w+') as res_f:
    #             res_f.write('X Y '
    #                         'height_x center_x width_x '
    #                         'height_y center_y width_y '
    #                         'height_z center_z width_z '
    #                         '|q| status\n')
    #             x_height, x_center, x_width = self.x_fit
    #             y_height, y_center, y_width = self.y_fit
    #             z_height, z_center, z_width = self.z_fit
    #             q = np.sqrt(x_center ** 2 +
    #                         y_center ** 2 +
    #                         z_center ** 2)
    #             status = self.status
    #             x, y = self.scan_positions
    #
    #             for i, s in enumerate(status):
    #                 r = [x[i], y[i],
    #                      x_height[i], x_center[i], x_width[i],
    #                      y_height[i], y_center[i], y_width[i],
    #                      z_height[i], z_center[i], z_width[i],
    #                      q[i], s]
    #                 res_str = '{0}\n'.format(' '.join(str(e) for e in r))
    #                 res_f.write(res_str)

#
# class FitH5Writer(FitH5):
#
#     def __init__(self, h5_f, mode='a', **kwargs):
#         super(FitH5Writer, self).__init__(h5_f, mode=mode, **kwargs)
#
#     def set_scan_positions(self, x, y):
#         path_tpl = FitH5.scan_positions_path + '/{0}'
#         self._set_array_data(path_tpl.format('x'), x)
#         self._set_array_data(path_tpl.format('y'), y)
#
#     def set_x_fit(self, height, center, width):
#         self.__set_fit('x', height, center, width)
#
#     def set_y_fit(self, height, center, width):
#         self.__set_fit('y', height, center, width)
#
#     def set_z_fit(self, height, center, width):
#         self.__set_fit('z', height, center, width)
#
#     def __set_fit(self, axis, height, center, width):
#         path_tpl = FitH5.q_fit_path.format(axis) + '/{0}'
#         self._set_array_data(path_tpl.format('height'), height)
#         self._set_array_data(path_tpl.format('center'), center)
#         self._set_array_data(path_tpl.format('width'), width)
#
#     def set_status(self, status):
#         self._set_array_data(FitH5.status_path, status)
#
#     def __set_axis_values(self, axis, values):
#         self._set_array_data(FitH5.axis_path.format(axis), values)
#
#     def set_x_axis(self, values):
#         self.__set_axis_values('x', values)
#
#     def set_y_axis(self, values):
#         self.__set_axis_values('y', values)
#
#     def set_z_axis(self, values):
#         self.__set_axis_values('z', values)


# class FitH5(XsocsH5Base):
#     scan_positions_path = 'scan_positions'
#     q_fit_path = 'q{0}fit'
#     status_path = 'success'
#     axis_path = 'axis/{0}'
#
#     @property
#     def scan_positions(self):
#         with self:
#             return self.sample_x, self.sample_y
#
#     sample_x = property(lambda self:
#                         self._get_array_data(FitH5.scan_positions_path + '/x'))
#
#     sample_y = property(lambda self:
#                         self._get_array_data(FitH5.scan_positions_path + '/y'))
#
#     x_fit = property(lambda self: self.__get_fit('x'))
#
#     y_fit = property(lambda self: self.__get_fit('y'))
#
#     z_fit = property(lambda self: self.__get_fit('z'))
#
#     x_height = property(lambda self: self.__get_data('x', 'height'))
#
#     y_height = property(lambda self: self.__get_data('y', 'height'))
#
#     z_height = property(lambda self: self.__get_data('z', 'height'))
#
#     x_center = property(lambda self: self.__get_data('x', 'center'))
#
#     y_center = property(lambda self: self.__get_data('y', 'center'))
#
#     z_center = property(lambda self: self.__get_data('z', 'center'))
#
#     x_width = property(lambda self: self.__get_data('x', 'width'))
#
#     y_width = property(lambda self: self.__get_data('y', 'width'))
#
#     z_width = property(lambda self: self.__get_data('z', 'width'))
#
#     status = property(lambda self: self._get_array_data(FitH5.status_path))
#
#     x_axis = property(lambda self: self.__get_axis_values('x'))
#
#     y_axis = property(lambda self: self.__get_axis_values('y'))
#
#     z_axis = property(lambda self: self.__get_axis_values('z'))
#
#     def __get_fit(self, axis):
#         with self:
#             height = self.__get_data(axis, 'height')
#             center = self.__get_data(axis, 'center')
#             width = self.__get_data(axis, 'width')
#         return height, center, width
#
#     def __get_data(self, axis, data):
#         data_path = FitH5.q_fit_path.format(axis) + '/{0}'.format(data)
#         return self._get_array_data(data_path)
#
#     def __get_axis_values(self, axis):
#         return self._get_array_data(FitH5.axis_path.format(axis))
#
#     def export_txt(self, filename):
#         with self:
#             with open(filename, 'w+') as res_f:
#                 res_f.write('X Y '
#                             'height_x center_x width_x '
#                             'height_y center_y width_y '
#                             'height_z center_z width_z '
#                             '|q| status\n')
#                 x_height, x_center, x_width = self.x_fit
#                 y_height, y_center, y_width = self.y_fit
#                 z_height, z_center, z_width = self.z_fit
#                 q = np.sqrt(x_center ** 2 +
#                             y_center ** 2 +
#                             z_center ** 2)
#                 status = self.status
#                 x, y = self.scan_positions
#
#                 for i, s in enumerate(status):
#                     r = [x[i], y[i],
#                          x_height[i], x_center[i], x_width[i],
#                          y_height[i], y_center[i], y_width[i],
#                          z_height[i], z_center[i], z_width[i],
#                          q[i], s]
#                     res_str = '{0}\n'.format(' '.join(str(e) for e in r))
#                     res_f.write(res_str)
#
#
# class FitH5Writer(FitH5):
#
#     def __init__(self, h5_f, mode='a', **kwargs):
#         super(FitH5Writer, self).__init__(h5_f, mode=mode, **kwargs)
#
#     def set_scan_positions(self, x, y):
#         path_tpl = FitH5.scan_positions_path + '/{0}'
#         self._set_array_data(path_tpl.format('x'), x)
#         self._set_array_data(path_tpl.format('y'), y)
#
#     def set_x_fit(self, height, center, width):
#         self.__set_fit('x', height, center, width)
#
#     def set_y_fit(self, height, center, width):
#         self.__set_fit('y', height, center, width)
#
#     def set_z_fit(self, height, center, width):
#         self.__set_fit('z', height, center, width)
#
#     def __set_fit(self, axis, height, center, width):
#         path_tpl = FitH5.q_fit_path.format(axis) + '/{0}'
#         self._set_array_data(path_tpl.format('height'), height)
#         self._set_array_data(path_tpl.format('center'), center)
#         self._set_array_data(path_tpl.format('width'), width)
#
#     def set_status(self, status):
#         self._set_array_data(FitH5.status_path, status)
#
#     def __set_axis_values(self, axis, values):
#         self._set_array_data(FitH5.axis_path.format(axis), values)
#
#     def set_x_axis(self, values):
#         self.__set_axis_values('x', values)
#
#     def set_y_axis(self, values):
#         self.__set_axis_values('y', values)
#
#     def set_z_axis(self, values):
#         self.__set_axis_values('z', values)
