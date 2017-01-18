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


class FitH5QAxis(object):
    axis_values = range(3)
    qx_axis, qy_axis, qz_axis = axis_values
    axis_names = ('qx', 'qy', 'qz')

    @staticmethod
    def axis_name(axis):
        return FitH5QAxis.axis_names[axis]


class FitH5(XsocsH5Base):
    """
    File containing fit results.
    Requirements :
    - the number of sample position is defined at entry level : all processes
    within the same entry are applied to the same sample points.
    - all results arrays within an entry (even if they don't belong to the same
    process) have the same size (equal to the number of sample points defined
    for that entry)
    - all arrays are 1D.
    """
    # _axis_values = range(3)
    # qx_axis, qy_axis, qz_axis = _axis_values
    # axis_names = ('qx', 'qy', 'qz')

    title_path = '{entry}/title'
    start_time_path = '{entry}/start_time'
    end_time_path = '{entry}/end_time'
    date_path = '{entry}/{process}/date'
    qspace_axis_path = '{entry}/qspace_axis/{axis}'
    status_path = '{entry}/{process}/status/{axis}'
    configuration_path = '{entry}/{process}/configuration'
    result_grp_path = '{entry}/{process}/results'
    result_path = '{entry}/{process}/results/{result}/{axis}'
    scan_x_path = '{entry}/sample/x_pos'
    scan_y_path = '{entry}/sample/y_pos'

    def title(self, entry):
        """
        Returns the title for the given entry.
        :param entry:
        :return:
        """
        with self._get_file() as h5_file:
            path = entry + '/title'
            return h5_file[path][()]

    def entries(self):
        """
        Return the entry names.
        :return:
        """
        with self._get_file() as h5_file:
            # TODO : this isnt pretty but for some reason the attrs.get() fails
            # when there is no attribute NX_class (should return the default
            # None)
            return sorted([key for key in h5_file
                           if ('NX_class' in h5_file[key].attrs and
                               h5_file[key].attrs[
                                   'NX_class'] == 'NXentry')])

    def processes(self, entry):
        """
        Return the processes names for the given entry.
        :param entry:
        :return:
        """
        with self._get_file() as h5_file:
            entry_grp = h5_file[entry]
            processes = sorted([key for key in entry_grp
                                if ('NX_class' in entry_grp[key].attrs and
                                    entry_grp[key].attrs[
                                        'NX_class'] == 'NXprocess')])
        return processes

    def get_result_names(self, entry, process):
        """
        Returns the result names for the given process. Names are ordered
        alphabetically.
        :param entry:
        :param process:
        :return:
        """
        results_path = self.result_grp_path.format(entry=entry,
                                                   process=process)
        with self._get_file() as h5_file:
            return sorted(h5_file[results_path].keys())

    def get_status(self, entry, process, axis):
        """
        Returns the fit status for the given entry/process/axis
        :param entry:
        :param process:
        :param axis: FitH5QAxis.qx_axis, FitH5QAxis.qy_axis
         or FitH5QAxis.qz_axis
        :return:
        """
        axis_name = FitH5QAxis.axis_name(axis)
        status_path = FitH5.status_path.format(entry=entry,
                                               process=process,
                                               axis=axis_name)
        return self._get_array_data(status_path)

    def scan_x(self, entry):
        """
        Return the sample points coordinates along x for the given entry.
        :param entry:
        :return:
        """
        dset_path = FitH5.scan_x_path.format(entry=entry)
        return self._get_array_data(dset_path)

    def scan_y(self, entry):
        """
        Return the sample points coordinates along y for the given entry.
        :param entry:
        :return:
        """
        dset_path = FitH5.scan_y_path.format(entry=entry)
        return self._get_array_data(dset_path)

    def get_qx(self, entry):
        """
        Returns the axis values for qx for the given entry.
        :param entry:
        :return:
        """
        return self.__get_axis_values(entry, FitH5QAxis.qx_axis)

    def get_qy(self, entry):
        """
        Returns the axis values for qy for the given entry.
        :param entry:
        :return:
        """
        return self.__get_axis_values(entry, FitH5QAxis.qy_axis)

    def get_qz(self, entry):
        """
        Returns the axis values for qz for the given entry.
        :param entry:
        :return:
        """
        return self.__get_axis_values(entry, FitH5QAxis.qz_axis)

    def __get_axis_values(self, entry, axis):
        """
        Returns the axis values.
        :param entry:
        :param axis: FitH5QAxis.qx_axis, FitH5QAxis.qy_axis
         or FitH5QAxis.qz_axis
        :return:
        """
        axis_name = FitH5QAxis.axis_name(axis)
        return self._get_array_data(FitH5.qspace_axis_path.format(
            entry=entry, axis=axis_name))

    def get_axis_result(self, entry, process, result, axis):
        """
        Returns the results for the given entry/process/result name/axis.
        :param entry:
        :param process:
        :param result:
        :param axis: FitH5QAxis.qx_axis, FitH5QAxis.qy_axis or
         FitH5QAxis.qz_axis
        :return:
        """
        assert axis in FitH5QAxis.axis_values
        axis_name = FitH5QAxis.axis_name(axis)
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=result,
                                               axis=axis_name)
        return self._get_array_data(result_path)

    def get_qx_result(self, entry, process, result):
        """
        Returns the results (qx) for the given entry/process/result name.
        :param entry:
        :param process:
        :param result:
        :return:
        """
        return self.get_axis_result(entry, process, result, FitH5QAxis.qx_axis)

    def get_qy_result(self, entry, process, result):
        """
        Returns the results (qy) for the given entry/process/result name.
        :param entry:
        :param process:
        :param result:
        :return:
        """
        return self.get_axis_result(entry, process, result, FitH5QAxis.qy_axis)

    def get_qz_result(self, entry, process, result):
        """
        Returns the results (qz) for the given entry/process/result name.
        :param entry:
        :param process:
        :param result:
        :return:
        """
        return self.get_axis_result(entry, process, result, FitH5QAxis.qz_axis)

    def get_result(self, entry, process, result):
        """
        Returns the results values (qx, qy, qz) for
        the given entry/process/result name.
        :param entry:
        :param process:
        :param result:
        :return: a FitResult instance.
        """
        with self:
            results = {}
            for axis in FitH5QAxis.axis_values:
                results[FitH5QAxis.axis_name(axis)] = \
                    self.get_axis_result(entry, process, result, axis)
            return FitResult(name=result, **results)

    def get_n_points(self, entry):
        """
        Returns the number of sample positions for this entry.
        :param entry:
        :return:
        """
        dset_path = FitH5.scan_x_path.format(entry=entry)
        shape = self._get_array_data(dset_path, shape=True)
        return shape[0]

    def export_csv(self, entry, filename):
        """
        Exports an entry results as csv.
        :param entry:
        :param filename:
        :return:
        """

        x, y = self.scan_x(entry), self.scan_y(entry)

        processes = self.processes(entry)

        if len(processes) == 0:
            raise ValueError('No process found for entry {0}.'.format(entry))

        # with open(filename, 'w+') as res_f:
        with self:

            header_process = ['_', 'process:']
            header_list = ['X', 'Y']
            for process in processes:
                result_names = self.get_result_names(entry, process)
                for axis in FitH5QAxis.axis_names:
                    for result_name in result_names:
                        header_process.append(process)
                        header_list.append(result_name + '_' + axis)
                    header_process.append(process)
                    header_list.append('status_' + axis)

            header = ' '.join(header_process) + '\n' + ' '.join(header_list)

            results = np.zeros((len(x), len(header_list)))

            results[:, 0] = x
            results[:, 1] = y

            col_idx = 2
            for process in processes:
                result_names = self.get_result_names(entry, process)
                for axis in FitH5QAxis.axis_values:
                    for result_name in result_names:
                        result = self.get_axis_result(entry,
                                                      process,
                                                      result_name,
                                                      axis)
                        results[:, col_idx] = result
                        col_idx += 1
                    results[:, col_idx] = self.get_status(entry,
                                                          process,
                                                          axis)
                    col_idx += 1

            np.savetxt(filename,
                       results,
                       fmt='%.10g',
                       header=header,
                       comments='')


class FitH5Writer(FitH5):

    def create_entry(self, entry):
        with self._get_file() as h5_file:
            entries = self.entries()
            if len(entries) > 0:
                raise ValueError('FitH5 doesnt support multiple entries '
                                 'yet.')
            # TODO : check if it already exists
            entry_grp = h5_file.require_group(entry)
            entry_grp.attrs['NX_class'] = np.string_('NXentry')

    def create_process(self, entry, process):
        # TODO : check that there isn't already an existing process
        with self._get_file() as h5_file:

            processes = self.processes(entry)
            if len(processes) > 0:
                raise ValueError('FitH5 doesnt support multiple processes '
                                 'yet.')

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

    def set_status(self, entry, process, axis, data):
        axis_name = FitH5QAxis.axis_name(axis)
        status_path = FitH5.status_path.format(entry=entry,
                                               process=process,
                                               axis=axis_name)
        self._set_array_data(status_path, data)

    def __set_axis_result(self, entry, process, name, q_axis, data):
        assert q_axis in FitH5QAxis.axis_values
        axis_name = FitH5QAxis.axis_name(q_axis)
        result_path = FitH5.result_path.format(entry=entry,
                                               process=process,
                                               result=name,
                                               axis=axis_name)
        self._set_array_data(result_path, data)

    def set_qx_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5QAxis.qx_axis, data)

    def set_qy_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5QAxis.qy_axis, data)

    def set_qz_result(self, entry, process, name, data):
        self.__set_axis_result(entry, process, name, FitH5QAxis.qz_axis, data)

    def __set_axis_values(self, entry, axis, values):
        axis_name = FitH5QAxis.axis_name(axis)
        self._set_array_data(FitH5.qspace_axis_path.format(entry=entry,
                                                           axis=axis_name),
                             values)

    def set_qx(self, entry, values):
        self.__set_axis_values(entry, FitH5QAxis.qx_axis, values)

    def set_qy(self, entry, values):
        self.__set_axis_values(entry, FitH5QAxis.qy_axis, values)

    def set_qz(self, entry, values):
        self.__set_axis_values(entry, FitH5QAxis.qz_axis, values)
