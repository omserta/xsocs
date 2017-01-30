#!/usr/bin/python
# coding: utf8
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
__date__ = "01/06/2016"
__license__ = "MIT"


from collections import OrderedDict

import numpy as np


class FitStatus(object):
    """
    Enum for the fit status
    Starting at 1 for compatibility reasons.
    """
    OK, FAILED = range(1, 3)


class FitResult(object):
    """
    Fit results
    """
    _AXIS = QX_AXIS, QY_AXIS, QZ_AXIS = range(3)
    _AXIS_NAMES = ('qx', 'qy', 'qz')

    def __init__(self, entry,
                 q_x, q_y, q_z,
                 sample_x, sample_y):
        super(FitResult, self).__init__()

        self._entry = entry

        self._sample_x = sample_x
        self._sample_y = sample_y

        self._q_x = q_x
        self._q_y = q_y
        self._q_z = q_z

        self._processes = OrderedDict()

        n_pts = len(sample_x)

        self._status = OrderedDict([('qx_status', np.zeros(n_pts)),
                                    ('qy_status', np.zeros(n_pts)),
                                    ('qz_status', np.zeros(n_pts))])

    def processes(self):
        """
        Returns the process names
        :return:
        """
        return self._processes.keys()

    def params(self, process):
        return self._get_process(process, create=False)['params'].keys()

    def status(self, axis):
        """
        Returns the status for the given axis.
        :param axis:
        :return:
        """
        assert axis in self._AXIS

        return self._status[self._AXIS_NAMES[axis]][:]

    def qx_status(self):
        """
        Returns qx fit status
        :return:
        """
        return self.status(self.QX_AXIS)

    def qy_status(self):
        """
        Returns qy fit status
        :return:
        """
        return self.status(self.QY_AXIS)

    def qz_status(self):
        """
        Returns qz fit status
        :return:
        """
        return self.status(self.QZ_AXIS)

    def results(self, process, param, axis=None):
        """
        Returns the fitted parameter results for a given process.
        :param process: process name
        :param param: param name
        :param axis: if provided, returns only the result for the given axis
        :return:
        """

        param = self._get_param(process, param, create=False)

        if axis is not None:
            assert axis in self._AXIS
            return param[self._AXIS_NAMES[axis]]
        return param

    def qx_results(self, process, param):
        """
        Returns qx fit results for the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.results(process, param, axis=self.QX_AXIS)

    def qy_results(self, process, param):
        """
        Returns qy fit results for the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.results(process, param, axis=self.QY_AXIS)

    def qz_results(self, process, param):
        """
        Returns qz fit results for the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.results(process, param, axis=self.QZ_AXIS)

    def add_qx_result(self, process, param, result):
        self._add_axis_result(process, self.QX_AXIS, param, result)

    def add_qy_result(self, process, param, result):
        self._add_axis_result(process, self.QY_AXIS, param, result)

    def add_qz_result(self, process, param, result):
        self._add_axis_result(process, self.QZ_AXIS, param, result)

    def _add_axis_result(self, process, axis, param, result):
        assert axis in self._AXIS

        param_data = self._get_param(process, param)
        param_data[self._AXIS_NAMES[axis]] = result

    def set_qx_status(self, status):
        self._set_axis_status(self.QX_AXIS, status)

    def set_qy_status(self, status):
        self._set_axis_status(self.QY_AXIS, status)

    def set_qz_status(self, status):
        self._set_axis_status(self.QZ_AXIS, status)

    def _set_axis_status(self, axis, status):
        assert axis in self._AXIS

        self._status[self._AXIS_NAMES[axis]] = status

    def _get_process(self, process, create=True):

        if process not in self._processes:
            if not create:
                raise KeyError('Unknown process {0}.'.format(process))
            _process = OrderedDict([('params', OrderedDict())])
            self._processes[process] = _process
        else:
            _process = self._processes[process]

        return _process

    def _get_param(self, process, param, create=True):
        process = self._get_process(process, create=create)
        params = process['params']

        if param not in params:
            if not create:
                raise KeyError('Unknown param {0}.'.format(param))
            _param = OrderedDict()
            for axis in self._AXIS_NAMES:
                _param[axis] = None
            params[param] = _param
        else:
            _param = params[param]

        return _param

    entry = property(lambda self: self._entry)

    sample_x = property(lambda self: self._sample_x)
    sample_y = property(lambda self: self._sample_y)

    q_x = property(lambda self: self._q_x)
    q_y = property(lambda self: self._q_y)
    q_z = property(lambda self: self._q_z)


if __name__ == '__main__':
    pass
