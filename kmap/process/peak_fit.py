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

import time
import ctypes
import multiprocessing as mp
from collections import namedtuple, OrderedDict
import multiprocessing.sharedctypes as mp_sharedctypes

import numpy as np

# from silx.math import curve_fit
from ..io import QSpaceH5
from .fit_funcs import gaussian_fit, centroid

disp_times = False


class FitTypes(object):
    ALLOWED = range(2)
    LEASTSQ, CENTROID = ALLOWED


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

    def processes(self):
        """
        Returns the process names
        :return:
        """
        return self._processes.keys()

    def params(self, process):
        return self._get_process(process, create=False)['params'].keys()

    def status(self, process, axis):
        assert axis in self._AXIS

        process = self._get_process(process, create=False)

        return process['status'][self._AXIS_NAMES[axis]]

    def qx_status(self, process):
        """
        Returns qx fit status the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.status(process, self.QX_AXIS)

    def qy_status(self, process):
        """
        Returns qy fit status the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.status(process, self.QY_AXIS)

    def qz_status(self, process):
        """
        Returns qz fit status the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.status(process, self.QZ_AXIS)

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

    def qy_results(self, process):
        """
        Returns qy fit results for the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.results(process, axis=self.QY_AXIS)

    def qz_results(self, process):
        """
        Returns qz fit results for the given process
        :param process:
        :param param: param name
        :return:
        """
        return self.results(process, axis=self.QZ_AXIS)

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
        
    def set_qx_status(self, process, status):
        self._set_axis_status(process, self.QX_AXIS, status)

    def set_qy_status(self, process, status):
        self._set_axis_status(process, self.QY_AXIS, status)

    def set_qz_status(self, process, status):
        self._set_axis_status(process, self.QZ_AXIS, status)

    def _set_axis_status(self, process, axis, status):
        assert axis in self._AXIS

        _process = self._get_process(process)
        statuses = _process['status']

        statuses[self._AXIS_NAMES[axis]] = status

    def _get_process(self, process, create=True):

        if process not in self._processes:
            if not create:
                raise KeyError('Unknown process {0}.'.format(process))
            status = OrderedDict([('qx_status', None),
                                  ('qy_status', None),
                                  ('qz_status', None)])
            _process = OrderedDict([('params', OrderedDict()),
                                    ('status', status)])
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

    q_x = property(lambda  self: self._q_x)
    q_y = property(lambda  self: self._q_y)
    q_z = property(lambda self: self._q_z)


class FitSharedResults(object):
    def __init__(self,
                 n_points=None,
                 n_params=None,
                 shared_results=None,
                 shared_status=None):
        super(FitSharedResults, self).__init__()
        assert n_points is not None, n_params is not None

        self._shared_qx_results = None
        self._shared_qy_results = None
        self._shared_qz_results = None

        self._shared_qx_status = None
        self._shared_qy_status = None
        self._shared_qz_status = None

        self._npy_qx_results = None
        self._npy_qy_results = None
        self._npy_qz_results = None

        self._npy_qx_status = None
        self._npy_qy_status = None
        self._npy_qz_status = None

        self._n_points = n_points
        self._n_params = n_params

        self._init_shared_results(shared_results)
        self._init_shared_status(shared_status)
        self._init_npy_results()
        self._init_npy_status()

    def _init_shared_results(self, shared_results=None):
        if shared_results is None:
            self._shared_qx_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
            self._shared_qy_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
            self._shared_qz_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
        else:
            self._shared_qx_results = shared_results[0]
            self._shared_qy_results = shared_results[1]
            self._shared_qz_results = shared_results[2]

    def _init_shared_status(self, shared_status=None):
        if shared_status is None:
            self._shared_qx_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
            self._shared_qy_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
            self._shared_qz_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
        else:
            self._shared_qx_status = shared_status[0]
            self._shared_qy_status = shared_status[1]
            self._shared_qz_status = shared_status[2]

    def _init_npy_results(self):
        self._npy_qx_results = np.frombuffer(self._shared_qx_results)
        self._npy_qx_results.shape = self._n_points, self._n_params
        self._npy_qy_results = np.frombuffer(self._shared_qy_results)
        self._npy_qy_results.shape = self._n_points, self._n_params
        self._npy_qz_results = np.frombuffer(self._shared_qz_results)
        self._npy_qz_results.shape = self._n_points, self._n_params

    def _init_npy_status(self):
        self._npy_qx_status = np.frombuffer(self._shared_qx_status,
                                            dtype=bool)
        self._npy_qy_status = np.frombuffer(self._shared_qy_status,
                                            dtype=bool)
        self._npy_qz_status = np.frombuffer(self._shared_qz_status,
                                            dtype=bool)

    def set_qx_results(self, idx, results, status):
        self._npy_qx_results[idx] = results
        self._npy_qx_status[idx] = status
        
    def set_qy_results(self, idx, results, status):
        self._npy_qy_results[idx] = results
        self._npy_qy_status[idx] = status
        
    def set_qz_results(self, idx, results, status):
        self._npy_qz_results[idx] = results
        self._npy_qz_status[idx] = status

    def local_copy(self):
        shared_results = (self._shared_qx_results,
                          self._shared_qy_results,
                          self._shared_qz_results)
        shared_status = (self._shared_qx_status,
                         self._shared_qy_status,
                         self._shared_qz_status)
        return FitSharedResults(n_points=self._n_points,
                                n_params=self._n_params,
                                shared_results=shared_results,
                                shared_status=shared_status)

    def fit_results(self, *args, **kwargs):
        raise NotImplementedError('')


class GaussianResults(FitSharedResults):
    def __init__(self,
                 n_points=None,
                 shared_results=None,
                 shared_status=None):
        super(GaussianResults, self).__init__(n_points=n_points,
                                              n_params=3,
                                              shared_results=shared_results,
                                              shared_status=shared_status)

    def fit_results(self, *args, **kwargs):
        qx_results = self._npy_qx_results
        qy_results = self._npy_qy_results
        qz_results = self._npy_qz_results

        qx_status = self._npy_qx_status
        qy_status = self._npy_qy_status
        qz_status = self._npy_qz_status

        fit_name = 'Gaussian'
        results = FitResult(fit_name, *args, **kwargs)
        results.add_qx_result('gaussian', 'intensity', qx_results[:, 0].ravel())
        results.add_qx_result('gaussian', 'position', qx_results[:, 1].ravel())
        results.add_qx_result('gaussian', 'width', qx_results[:, 2].ravel())
        results.set_qx_status('gaussian', qx_status)

        results.add_qy_result('gaussian', 'intensity', qy_results[:, 0].ravel())
        results.add_qy_result('gaussian', 'position', qy_results[:, 1].ravel())
        results.add_qy_result('gaussian', 'width', qy_results[:, 2].ravel())
        results.set_qy_status('gaussian', qy_status)

        results.add_qz_result('gaussian', 'intensity', qz_results[:, 0].ravel())
        results.add_qz_result('gaussian', 'position', qz_results[:, 1].ravel())
        results.add_qz_result('gaussian', 'width', qz_results[:, 2].ravel())
        results.set_qz_status('gaussian', qz_status)

        return results


# class CentroidResults(FitSharedResults):
#     def __init__(self,
#                  n_points=None,
#                  shared_results=None,
#                  shared_status=None):
#         super(CentroidResults, self).__init__(n_points=n_points,
#                                               n_params=2,
#                                               shared_results=shared_results,
#                                               shared_status=shared_status)
#
#     def fit_results(self):
#         qx_results = self._npy_qx_results
#         qy_results = self._npy_qy_results
#         qz_results = self._npy_qz_results
#
#         qx_status = self._npy_qx_status
#         qy_status = self._npy_qy_status
#         qz_status = self._npy_qz_status
#
#         fit_name = 'Centroid'
#         q_x_results = {'height': qx_results[:, 0].ravel(),
#                        'position': qx_results[:, 1].ravel()}
#         q_y_results = {'height': qy_results[:, 0].ravel(),
#                        'position': qy_results[:, 1].ravel()}
#         q_z_results = {'height': qz_results[:, 0].ravel(),
#                        'position': qz_results[:, 1].ravel()}
#
#         return [(fit_name, q_x_results, q_y_results, q_z_results)]


def peak_fit(qspace_f,
             fit_type=FitTypes.LEASTSQ,
             indices=None,
             n_proc=None,
             roiIndices=None):
    """
    :param qspace_f: path to the HDF5 file containing the qspace cubes
    :type data_h5f: `str`

    :param fit_type:
    :type img_indices: *optional*

    :param indices: indices of the cubes (in the input HDF5 dataset) for which
        the qx/qy/qz peaks coordinates will be computed. E.g : if the array
        [1, 2, 3] is provided, only those cubes will be fitted.
    :type img_indices: *optional* `array_like`

    :param n_proc: number of process to use. If None, the number of process
        used will be the one returned by multiprocessing.cpu_count().
    :type n_proc: `int`
    """

    t_total = time.time()

    if fit_type not in FitTypes.ALLOWED:
        raise ValueError('Unknown fit type : {0}'.format(fit_type))

    if fit_type == FitTypes.LEASTSQ:
        fit_fn = gaussian_fit
        n_params = 9
    if fit_type == FitTypes.CENTROID:
        fit_fn = centroid
        n_params = 9

    with QSpaceH5.QSpaceH5(qspace_f) as qspace_h5:
        with qspace_h5.qspace_dset_ctx() as dset:
            qdata_shape = dset.shape

        n_points = qdata_shape[0]

        if indices is None:
            indices = range(n_points)

        n_indices = len(indices)

        x_pos = qspace_h5.sample_x[indices]
        y_pos = qspace_h5.sample_y[indices]

        # shared_res = mp_sharedctypes.RawArray(ctypes.c_double, n_indices * 9)
        # # TODO : find something better
        # shared_success = mp_sharedctypes.RawArray(ctypes.c_bool, n_indices)
        # # success = np.ndarray((n_indices,), dtype=np.bool)
        # # success[:] = True

    shared_results = GaussianResults(n_points=n_indices)

    # with h5py.File(qspace_f, 'r') as qspace_h5:
    #
    #     q_x = qspace_h5['bins_edges/x'][:]
    #     q_y = qspace_h5['bins_edges/y'][:]
    #     q_z = qspace_h5['bins_edges/z'][:]
    #     qdata = qspace_h5['data/qspace']
    #
    #     n_points = qdata.shape[0]
    #
    #     if indices is None:
    #         indices = range(n_points)
    #
    #     n_indices = len(indices)
    #
    #     x_pos = qspace_h5['geom/x'][indices]
    #     y_pos = qspace_h5['geom/y'][indices]
    #
    #     shared_res = mp_sharedctypes.RawArray(ctypes.c_double, n_indices * 9)
    #     # TODO : find something better
    #     shared_success = mp_sharedctypes.RawArray(ctypes.c_bool, n_indices)
    #     success = np.ndarray((n_indices,), dtype=np.bool)
    #     success[:] = True
    #
    #     # this has to be done otherwise h5py complains about not being
    #     # able to open compressed datasets from other processes
    #     del qdata

    # results = np.ndarray((n_indices, 11), dtype=np.double)
    # results[:, 0] = x_pos
    # results[:, 1] = y_pos

    manager = mp.Manager()

    read_lock = manager.Lock()
    idx_queue = manager.Queue()

    if n_proc is None:
        n_proc = mp.cpu_count()

    pool = mp.Pool(n_proc,
                   initializer=_init_thread,
                   initargs=(shared_results,
                             fit_fn,
                             (n_indices, 9),
                             idx_queue,
                             qspace_f,
                             read_lock))
                   # initargs=(shared_res,
                   #           shared_success,
                   #           fit_fn,
                   #           (n_indices, 9),
                   #           idx_queue,
                   #           qspace_f,
                   #           read_lock))

    if disp_times:
        class myTimes(object):
            def __init__(self):
                self.t_read = 0.
                self.t_mask = 0.
                self.t_fit = 0.
                self.t_write = 0.

            def update(self, arg):
                (t_read_, t_mask_, t_fit_, t_write_) = arg
                self.t_read += t_read_
                self.t_mask += t_mask_
                self.t_fit += t_fit_
                self.t_write += t_write_

        res_times = myTimes()
        callback = res_times.update
    else:
        callback = None

    # creating the processes
    res_list = []
    for th_idx in range(n_proc):
        arg_list = (th_idx, roiIndices)
        res = pool.apply_async(_fit_process, args=arg_list, callback=callback)
        res_list.append(res)

    # sending the image indices
    for i_cube in indices:
        idx_queue.put(i_cube)

    # sending the None value to let the threads know that they should return
    for th_idx in range(n_proc):
        idx_queue.put(None)

    pool.close()
    pool.join()

    # fit_x = np.ndarray((n_indices, 3), dtype=np.float64)
    # fit_y = np.ndarray((n_indices, 3), dtype=np.float64)
    # fit_z = np.ndarray((n_indices, 3), dtype=np.float64)

    # results_np = np.frombuffer(shared_res)
    # results_np.shape = n_indices, 9
    # success = np.frombuffer(shared_success, dtype=bool)

    # results_np = shared_results._shared_array
    # success = shared_results._shared_status
    # qx_results = shared_results._shared_qx_results
    # qy_results = shared_results._shared_qy_results
    # qz_results = shared_results._shared_qz_results
    #
    # qx_status = shared_results._shared_qx_status
    # qy_status = shared_results._shared_qy_status
    # qz_status = shared_results._shared_qz_status
    

    # results[:, 2:5] = results_np[:, 0:3]
    # results[:, 5:8] = results_np[:, 3:6]
    # results[:, 8:11] = results_np[:, 6:9]

    t_total = time.time() - t_total
    if disp_times:
        print('Total : {0}.'.format(t_total))
        print('Read {0}'.format(res_times.t_read))
        print('Mask {0}'.format(res_times.t_mask))
        print('Fit {0}'.format(res_times.t_fit))
        print('Write {0}'.format(res_times.t_write))

    # status = np.zeros(x_pos.shape)
    # status[success] = 1

    with QSpaceH5.QSpaceH5(qspace_f) as qspace_h5:
        q_x = qspace_h5.qx
        q_y = qspace_h5.qy
        q_z = qspace_h5.qz

    if roiIndices is not None:
        xSlice = slice(roiIndices[0][0], roiIndices[0][1], 1)
        ySlice = slice(roiIndices[1][0], roiIndices[1][1], 1)
        zSlice = slice(roiIndices[2][0], roiIndices[2][1], 1)
        q_x = q_x[xSlice]
        q_y = q_y[ySlice]
        q_z = q_z[zSlice]

    # TODO : REFACTOR/IMPROVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if fit_type == FitTypes.LEASTSQ:
    #     fit_name = 'LeastSq'
    #     q_x_results = {'height': q_x_results[:, 0].ravel(),
    #                    'position': q_x_results[:, 1].ravel(),
    #                    'width': q_x_results[:, 2].ravel()}
    #     q_y_results = {'height': q_y_results[:, 3].ravel(),
    #                    'position': results_np[:, 4].ravel(),
    #                    'width': results_np[:, 5].ravel()}
    #     q_z_results = {'height': results_np[:, 6].ravel(),
    #                    'position': results_np[:, 7].ravel(),
    #                    'width': results_np[:, 8].ravel()}
    # else:
    #     fit_name = 'Centroid'
    #     q_x_results = {'height': results_np[:, 0].ravel(),
    #                    'position': results_np[:, 1].ravel()}
    #     q_y_results = {'height': results_np[:, 3].ravel(),
    #                    'position': results_np[:, 4].ravel()}
    #     q_z_results = {'height': results_np[:, 6].ravel(),
    #                    'position': results_np[:, 7].ravel()}

    # # TODO : REFACTOR/IMPROVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if fit_type == FitTypes.LEASTSQ:
    #     fit_name = 'LeastSq'
    #     q_x_results = {'height': results_np[:, 0].ravel(),
    #                    'position': results_np[:, 1].ravel(),
    #                    'width': results_np[:, 2].ravel()}
    #     q_y_results = {'height': results_np[:, 3].ravel(),
    #                    'position': results_np[:, 4].ravel(),
    #                    'width': results_np[:, 5].ravel()}
    #     q_z_results = {'height': results_np[:, 6].ravel(),
    #                    'position': results_np[:, 7].ravel(),
    #                    'width': results_np[:, 8].ravel()}
    # else:
    #     fit_name = 'Centroid'
    #     q_x_results = {'height': results_np[:, 0].ravel(),
    #                    'position': results_np[:, 1].ravel()}
    #     q_y_results = {'height': results_np[:, 3].ravel(),
    #                    'position': results_np[:, 4].ravel()}
    #     q_z_results = {'height': results_np[:, 6].ravel(),
    #                    'position': results_np[:, 7].ravel()}

    fit_results = shared_results.fit_results(sample_x=x_pos,
                                             sample_y=y_pos,
                                             q_x=q_x,
                                             q_y=q_y,
                                             q_z=q_z)

    # fit_results = FitResult(sample_x=x_pos,
    #                         sample_y=y_pos,
    #                         q_x=q_x,
    #                         q_y=q_y,
    #                         q_z=q_z,
    #                         q_x_results=q_x_results,
    #                         q_y_results=q_y_results,
    #                         q_z_results=q_z_results,
    #                         status=status,
    #                         fit_name=fit_name)
    return fit_results


def _init_thread(shared_res_,
                 fit_fn_,
                 result_shape_,
                 idx_queue_,
                 qspace_f_,
                 read_lock_):
    global shared_res, \
        shared_success, \
        fit_fn, \
        result_shape, \
        idx_queue, \
        qspace_f, \
        read_lock

    shared_res = shared_res_
    fit_fn = fit_fn_
    result_shape = result_shape_
    idx_queue = idx_queue_
    qspace_f = qspace_f_
    read_lock = read_lock_


def _gauss_first_guess(x, y):
    i_max = y.argmax()
    y_max = y[i_max]
    p1 = x[i_max]
    i_fwhm = np.where(y >= y_max / 2.)[0]
    fwhm = (x[1] - x[0]) * len(i_fwhm)
    p2 = fwhm / np.sqrt(2 * np.log(2))  # 2.35482
    p0 = y_max * np.sqrt(2 * np.pi) * p2
    return [p0, p1, p2]


def _fit_process(th_idx, roiIndices=None):
    print('Thread {0} started.'.format(th_idx))
    try:
        t_read = 0.
        t_fit = 0.
        t_mask = 0.

        # results = np.frombuffer(shared_res)
        # results.shape = result_shape
        # success = np.frombuffer(shared_success, dtype=bool)
        l_shared_res = shared_res.local_copy()
        # results = l_shared_res._shared_array
        # success = l_shared_res._shared_status

        qspace_h5 = QSpaceH5.QSpaceH5(qspace_f)

        # Put this in the main thread
        if roiIndices is not None:
            xSlice = slice(roiIndices[0][0], roiIndices[0][1], 1)
            ySlice = slice(roiIndices[1][0], roiIndices[1][1], 1)
            zSlice = slice(roiIndices[2][0], roiIndices[2][1], 1)

        # TODO : timeout to check if it has been canceled
        # read_lock.acquire()
        # with h5py.File(qspace_f, 'r') as qspace_h5:
        #     q_x = qspace_h5['bins_edges/x'][:]
        #     q_y = qspace_h5['bins_edges/y'][:]
        #     q_z = qspace_h5['bins_edges/z'][:]
        #     q_shape = qspace_h5['data/qspace'].shape
        #     q_dtype = qspace_h5['data/qspace'].dtype
        #     mask = np.where(qspace_h5['histo'][:] > 0)
        #     weights = qspace_h5['histo'][:][mask]
        with qspace_h5 as qspace_h5:
            q_x = qspace_h5.qx
            q_y = qspace_h5.qy
            q_z = qspace_h5.qz
            with qspace_h5.qspace_dset_ctx() as dset:
                q_shape = dset.shape
                q_dtype = dset.dtype
            histo = qspace_h5.histo

            if roiIndices:
                q_x = q_x[xSlice]
                q_y = q_y[ySlice]
                q_z = q_z[zSlice]
                histo = histo[xSlice, ySlice, zSlice]

            mask = np.where(histo > 0)
            weights = histo[mask]

        # read_lock.release()
        # print weights.max(), min(weights)
        read_cube = np.ascontiguousarray(np.zeros(q_shape[1:]),
                                         dtype=q_dtype)

        x_0 = None
        y_0 = None
        z_0 = None

        while True:
            # TODO : timeout
            i_cube = idx_queue.get()

            if i_cube is None:
                break

            if i_cube % 100 == 0:
                print(
                'Processing cube {0}/{1}.'.format(i_cube, result_shape[0]))

            t0 = time.time()
            # with h5py.File(qspace_f, 'r') as qspace_h5:
            #     qspace_h5['data/qspace'].read_direct(cube,
            #                                     source_sel=np.s_[i_cube],
            #                                     dest_sel=None)
            with qspace_h5.qspace_dset_ctx() as dset:
                dset.read_direct(read_cube,
                                 source_sel=np.s_[i_cube],
                                 dest_sel=None)
            t_read += time.time() - t0

            if roiIndices:
                cube = read_cube[xSlice, ySlice, zSlice]
            else:
                cube = read_cube

            t0 = time.time()
            cube[mask] = cube[mask] / weights
            t_mask = time.time() - t0

            t0 = time.time()

            success_x = True
            success_y = True
            success_z = True

            z_sum = cube.sum(axis=0).sum(axis=0)

            # if z_0 is None:
            # z_0 = _gauss_first_guess(q_z, z_sum)
            z_0 = [1.0, q_z.mean(), 1.0]

            try:
                fit_z = fit_fn(q_z, z_sum, z_0)
                z_0 = fit_z
            except Exception as ex:
                print('Z Failed', ex)
                z_0 = None
                fit_z = [np.nan, np.nan, np.nan]
                success_z = False

            l_shared_res.set_qz_results(i_cube, fit_z, success_z)

            z_sum = 0

            cube_sum_z = cube.sum(axis=2)

            y_sum = cube_sum_z.sum(axis=0)

            # if y_0 is None:
            # y_0 = _gauss_first_guess(q_y, y_sum)
            y_0 = [1.0, q_y.mean(), 1.0]

            try:
                fit_y = fit_fn(q_y, y_sum, y_0)
                y_0 = fit_y
            except Exception as ex:
                print('Y Failed', ex, i_cube)
                y_0 = None
                fit_y = [np.nan, np.nan, np.nan]
                success_y = False

            l_shared_res.set_qy_results(i_cube, fit_y, success_y)

            y_sum = 0

            x_sum = cube_sum_z.sum(axis=1)

            # if x_0 is None:
            # x_0 = _gauss_first_guess(q_x, x_sum)
            x_0 = [1.0, q_x.mean(), 1.0]

            try:
                fit_x = fit_fn(q_x, x_sum, x_0)
                x_0 = fit_x
            except Exception as ex:
                print('X Failed', ex)
                x_0 = None
                fit_x = [np.nan, np.nan, np.nan]
                success_x = False

            l_shared_res.set_qx_results(i_cube, fit_x, success_x)

            x_sum = 0

            t_fit += time.time() - t0

            t0 = time.time()

            # success[i_cube] = True
            #
            # if success_x:
            #     results[i_cube, 0:3] = fit_x
            # else:
            #     results[i_cube, 0:3] = np.nan
            #     success[i_cube] = False
            #
            # if success_y:
            #     results[i_cube, 3:6] = fit_y
            # else:
            #     results[i_cube, 3:6] = np.nan
            #     success[i_cube] = False
            #
            # if success_z:
            #     results[i_cube, 6:9] = fit_z
            # else:
            #     results[i_cube, 6:9] = np.nan
            #     success[i_cube] = False

            t_write = time.time() - t0

    except Exception as ex:
        print 'EX', ex

    times = (t_read, t_mask, t_fit, t_write)
    print('Thread {0} done ({1}).'.format(th_idx, times))
    return times
