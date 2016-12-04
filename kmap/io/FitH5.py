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

from collections import namedtuple

import numpy as np

from .XsocsH5Base import XsocsH5Base

FitResult = namedtuple('FitResult', ['name', 'qx', 'qy', 'qz'])


class FitH5(XsocsH5Base):
    _axis_values = range(3)
    qx_axis, qy_axis, qz_axis = _axis_values
    _axis_names = ('qx', 'qy', 'qz')

    title_path = '{entry}/title'
    start_time_path = '{entry}/start_time'
    end_time_path = '{entry}/end_time'
    date_path = '{entry}/{process}/date'
    qspace_axis_path = '{entry}/qspace_axis/{axis}'
    status_path = '{entry}/{process}/status'
    configuration_path = '{entry}/{process}/configuration'
    result_grp_path = '{entry}/{process}/results'
    result_path = '{entry}/{process}/results/{result}/{axis}'
    scan_x_path = '{entry}/sample/x_pos'
    scan_y_path = '{entry}/sample/y_pos'

    def title(self, entry):
        with self._get_file() as h5_file:
            path = entry + '/title'
            return h5_file[path][()]

    def entries(self):
        with self._get_file() as h5_file:
            # TODO : this isnt pretty but for some reason the attrs.get() fails
            # when there is no attribute NX_class (should return the default
            # None)
            return sorted([key for key in h5_file
                           if ('NX_class' in h5_file[key].attrs and
                               h5_file[key].attrs[
                                   'NX_class'] == 'NXentry')])

    def processes(self, entry):
        with self._get_file() as h5_file:
            entry_grp = h5_file[entry]
            processes = sorted([key for key in entry_grp
                                if ('NX_class' in entry_grp[key].attrs and
                                    entry_grp[key].attrs[
                                        'NX_class'] == 'NXprocess')])
        return processes

    def results(self, entry, process):
        with self._get_file() as h5_file:
            result_grp = h5_file[FitH5.result_grp_path.format(entry=entry,
                                                              process=process)]
            return sorted(result_grp.keys())

    def scan_x(self, entry):
        dset_path = FitH5.scan_x_path.format(entry=entry)
        return self._get_array_data(dset_path)

    def scan_y(self, entry):
        dset_path = FitH5.scan_y_path.format(entry=entry)
        return self._get_array_data(dset_path)

    def get_qx(self, entry):
        self.__get_axis_values(entry, FitH5.qx_axis)

    def get_qy(self, entry):
        self.__get_axis_values(entry, FitH5.qy_axis)

    def get_qz(self, entry):
        self.__get_axis_values(entry, FitH5.qz_axis)

    def __get_axis_values(self, entry, axis):
        axis_name = FitH5._axis_names[axis]
        self._get_array_data(FitH5.qspace_axis_path.format(entry=entry,
                                                           axis=axis_name))

    def result(self, entry, process, result):
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=result)
        return self._get_array_data(result_path)

    def __get_axis_result(self, entry, process, name, q_axis):
        assert q_axis in FitH5._axis_values
        axis_name = self._axis_names[q_axis]
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=name,
                                               axis=axis_name)
        return self._get_array_data(result_path)

    def get_qx_result(self, entry, process, result):
        return self.__get_axis_result(entry, process, result, FitH5.qx_axis)

    def get_qy_result(self, entry, process, result):
        return self.__get_axis_result(entry, process, result, FitH5.qy_axis)

    def get_qz_result(self, entry, process, result):
        return self.__get_axis_result(entry, process, result, FitH5.qz_axis)

    def get_result(self, entry, process, result):
        with self:
            results = {}
            for axis in FitH5._axis_values:
                results[FitH5._axis_names[axis]] = \
                    self.__get_axis_result(entry, process, result, axis)
            return FitResult(name=result, **results)


class FitH5Writer(FitH5):

    def create_entry(self, entry):
        with self._get_file() as h5_file:
            # TODO : check if it already exists
            entry_grp = h5_file.require_group(entry)
            entry_grp.attrs['NX_class'] = np.string_('NXentry')

    def create_process(self, entry, process):
        # TODO : check that there isn't already an existing process
        with self._get_file() as h5_file:
            if entry not in h5_file:
                self.create_entry(entry)
            entry_grp = h5_file[entry]

            # TODO : check if it exists
            process_grp = entry_grp.require_group(process)
            process_grp.attrs['NX_class'] = np.string_('NXprocess')
            results_grp = process_grp.require_group('results')
            results_grp.attrs['NX_class'] = np.string_('NXcollection')

    def set_scan_x(self, entry, x):
        dset_path = FitH5.scan_x_path.format(entry=entry)
        return self._set_array_data(dset_path, x)

    def set_scan_y(self, entry, y):
        dset_path = FitH5.scan_y_path.format(entry=entry)
        return self._set_array_data(dset_path, y)

    def set_title(self, entry, title):
        self._set_scalar_data(FitH5.title_path.format(entry), title)

    def set_status(self, entry, process, data):
        status_path = FitH5.status_path.format(entry=entry, process=process)
        self._set_array_data(status_path, data)

    def __set_axis_result(self, entry, process, name, q_axis, data):
        assert q_axis in FitH5._axis_values
        axis_name = self._axis_names[q_axis]
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=name,
                                               axis=axis_name)
        self._set_array_data(result_path, data)

    def set_qx_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5.qx_axis, data)

    def set_qy_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5.qy_axis, data)

    def set_qz_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5.qz_axis, data)

    def __set_axis_values(self, entry, axis, values):
        axis_name = FitH5._axis_names[axis]
        self._set_array_data(FitH5.qspace_axis_path.format(entry=entry,
                                                           axis=axis_name),
                             values)

    def set_qx(self, entry, values):
        self.__set_axis_values(entry, FitH5.qx_axis, values)

    def set_qy(self, entry, values):
        self.__set_axis_values(entry, FitH5.qy_axis, values)

    def set_qz(self, entry, values):
        self.__set_axis_values(entry, FitH5.qz_axis, values)

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
