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
from threading import Thread
import multiprocessing.sharedctypes as mp_sharedctypes

import numpy as np

# from silx.math import curve_fit
from ..io import QSpaceH5
from .fit_funcs import GaussianFitter, CentroidFitter, SilxFitter
from .sharedresults import (FitTypes, GaussianResults,
                            CentroidResults, SilxResults)
from .fitresults import FitStatus

disp_times = False


class PeakFitter(Thread):
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

    READY, RUNNING, DONE, ERROR, CANCELED = __STATUSES = range(5)

    def __init__(self,
                 qspace_f,
                 fit_type=FitTypes.GAUSSIAN,
                 n_peaks=1,
                 indices=None,
                 n_proc=None,
                 roi_indices=None):
        super(PeakFitter, self).__init__()

        self.__results = None
        self.__thread = None
        self.__progress = 0
        self.__callback = None

        self.__status = self.READY

        self.__indices = None

        self.__qspace_f = qspace_f
        self.__fit_type = fit_type
        self.__n_peaks = n_peaks

        if n_proc:
            self.__n_proc = n_proc
        else:
            n_proc = self.__n_proc = mp.cpu_count()

        self.__shared_progress = mp_sharedctypes.RawArray(ctypes.c_int32,
                                                          n_proc)

        if roi_indices is not None:
            self.__roi_indices = np.array(roi_indices[:])
        else:
            self.__roi_indices = None

        if fit_type not in FitTypes.ALLOWED:
            self.__set_status(self.ERROR)
            raise ValueError('Unknown fit type : {0}'.format(fit_type))

        try:
            with QSpaceH5.QSpaceH5(qspace_f) as qspace_h5:
                with qspace_h5.qspace_dset_ctx() as dset:
                    qdata_shape = dset.shape

                n_points = qdata_shape[0]

                if indices is None:
                    indices = range(n_points)
                else:
                    indices = indices[:]
        except IOError:
            self.__set_status(self.ERROR)
            raise

        self.__indices = np.array(indices)

    def __set_status(self, status):
        assert status in self.__STATUSES
        self.__status = status

    status = property(lambda self: self.__status)

    results = property(lambda self: self.__results)

    def peak_fit(self,
                 blocking=True,
                 callback=None):

        if self.__thread and self.__thread.is_alive():
            raise RuntimeError('A fit is already in progress.')

        self.__results = None

        if blocking:
            return self.__peak_fit()
        else:
            thread = self.__thread = Thread(target=self.__peak_fit)
            self.__callback = callback
            thread.start()

    def progress(self):
        return (100.0 *
                np.frombuffer(self.__shared_progress, dtype='int32').max() /
                (len(self.__indices) - 1))

    def __peak_fit(self):

        self.__set_status(self.RUNNING)

        qspace_f = self.__qspace_f
        fit_type = self.__fit_type
        indices = self.__indices
        n_proc = self.__n_proc
        roi_indices = self.__roi_indices
        shared_progress = self.__shared_progress
        n_peaks = self.__n_peaks

        t_total = time.time()

        progress = np.frombuffer(shared_progress, dtype='int32')
        progress[:] = 0

        n_indices = len(indices)

        try:
            with QSpaceH5.QSpaceH5(qspace_f) as qspace_h5:
                with qspace_h5:
                    x_pos = qspace_h5.sample_x[indices]
                    y_pos = qspace_h5.sample_y[indices]
        except IOError:
            self.__set_status(self.ERROR)
            raise

            # shared_res = mp_sharedctypes.RawArray(ctypes.c_double, n_indices * 9)
            # # TODO : find something better
            # shared_success = mp_sharedctypes.RawArray(ctypes.c_bool, n_indices)
            # # success = np.ndarray((n_indices,), dtype=np.bool)
            # # success[:] = True

        if fit_type == FitTypes.GAUSSIAN:
            fit_class = GaussianFitter
            n_peaks = n_peaks if n_peaks >= 1 else 1
            shared_results = GaussianResults(n_points=n_indices,
                                             n_peaks=n_peaks)
        elif fit_type == FitTypes.CENTROID:
            fit_class = CentroidFitter
            shared_results = CentroidResults(n_points=n_indices)
        elif fit_type == FitTypes.SILX:
            fit_class = SilxFitter
            n_peaks = n_peaks if n_peaks >= 1 else 1
            shared_results = SilxResults(n_points=n_indices,
                                         n_peaks=n_peaks)

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

        pool = mp.Pool(n_proc,
                       initializer=_init_thread,
                       initargs=(shared_results,
                                 shared_progress,
                                 fit_class,
                                 (n_indices, 9),
                                 idx_queue,
                                 qspace_f,
                                 read_lock))

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
            arg_list = (th_idx, roi_indices)
            res = pool.apply_async(_fit_process,
                                   args=arg_list,
                                   callback=callback)
            res_list.append(res)

        # sending the image indices
        for i_fit, i_cube in enumerate(indices):
            idx_queue.put([i_fit, i_cube])

        # sending the None value to let the threads know that they should return
        for th_idx in range(n_proc):
            idx_queue.put(None)

        pool.close()
        pool.join()

        t_total = time.time() - t_total
        if disp_times:
            print('Total : {0}.'.format(t_total))
            print('Read {0}'.format(res_times.t_read))
            print('Mask {0}'.format(res_times.t_mask))
            print('Fit {0}'.format(res_times.t_fit))
            print('Write {0}'.format(res_times.t_write))

        with QSpaceH5.QSpaceH5(qspace_f) as qspace_h5:
            q_x = qspace_h5.qx
            q_y = qspace_h5.qy
            q_z = qspace_h5.qz

        if roi_indices is not None:
            xSlice = slice(roi_indices[0][0], roi_indices[0][1], 1)
            ySlice = slice(roi_indices[1][0], roi_indices[1][1], 1)
            zSlice = slice(roi_indices[2][0], roi_indices[2][1], 1)
            q_x = q_x[xSlice]
            q_y = q_y[ySlice]
            q_z = q_z[zSlice]

        fit_results = shared_results.fit_results(sample_x=x_pos,
                                                 sample_y=y_pos,
                                                 q_x=q_x,
                                                 q_y=q_y,
                                                 q_z=q_z)

        self.__results = fit_results

        self.__set_status(self.DONE)

        if self.__callback:
            self.__callback()

        return fit_results


def _init_thread(shared_res_,
                 shared_prog_,
                 fit_class_,
                 result_shape_,
                 idx_queue_,
                 qspace_f_,
                 read_lock_):
    global shared_res, \
        shared_progress, \
        fit_class, \
        result_shape, \
        idx_queue, \
        qspace_f, \
        read_lock

    shared_res = shared_res_
    shared_progress = shared_prog_
    fit_class = fit_class_
    result_shape = result_shape_
    idx_queue = idx_queue_
    qspace_f = qspace_f_
    read_lock = read_lock_


def _fit_process(th_idx, roiIndices=None):
    print('Thread {0} started.'.format(th_idx))
    try:
        t_read = 0.
        t_fit = 0.
        t_mask = 0.

        l_shared_res = shared_res.local_copy()
        progress = np.frombuffer(shared_progress, dtype='int32')

        qspace_h5 = QSpaceH5.QSpaceH5(qspace_f)

        # Put this in the main thread
        if roiIndices is not None:
            xSlice = slice(roiIndices[0][0], roiIndices[0][1], 1)
            ySlice = slice(roiIndices[1][0], roiIndices[1][1], 1)
            zSlice = slice(roiIndices[2][0], roiIndices[2][1], 1)

        # TODO : timeout to check if it has been canceled
        # read_lock.acquire()
        with qspace_h5:
            q_x = qspace_h5.qx
            q_y = qspace_h5.qy
            q_z = qspace_h5.qz
            with qspace_h5.qspace_dset_ctx() as dset:
                q_shape = dset.shape
                q_dtype = dset.dtype
            histo = qspace_h5.histo

            if roiIndices is not None:
                q_x = q_x[xSlice]
                q_y = q_y[ySlice]
                q_z = q_z[zSlice]
                histo = histo[xSlice, ySlice, zSlice]

            mask = np.where(histo > 0)
            weights = histo[mask]

        # read_lock.release()
        read_cube = np.ascontiguousarray(np.zeros(q_shape[1:]),
                                         dtype=q_dtype)

        fitter = fit_class(q_x, q_y, q_z, l_shared_res)

        while True:
            # TODO : timeout
            next = idx_queue.get()

            if next is None:
                break

            i_fit, i_cube = next

            progress[th_idx] = i_fit

            if i_cube % 100 == 0:
                print(
                'Processing cube {0}/{1}.'.format(i_fit, result_shape[0]))

            t0 = time.time()
            with qspace_h5.qspace_dset_ctx() as dset:
                dset.read_direct(read_cube,
                                 source_sel=np.s_[i_cube],
                                 dest_sel=None)
            t_read += time.time() - t0

            if roiIndices is not None:
                cube = read_cube[xSlice, ySlice, zSlice]
            else:
                cube = read_cube

            t0 = time.time()
            cube[mask] = cube[mask] / weights
            t_mask = time.time() - t0

            t0 = time.time()

            z_sum = cube.sum(axis=0).sum(axis=0)
            cube_sum_z = cube.sum(axis=2)
            y_sum = cube_sum_z.sum(axis=0)
            x_sum = cube_sum_z.sum(axis=1)

            fitter.fit(i_fit, i_cube, x_sum, y_sum, z_sum)

            t_fit += time.time() - t0

            t0 = time.time()

            t_write = time.time() - t0

    except Exception as ex:
        print('EX', ex)

    times = (t_read, t_mask, t_fit, t_write)
    if disp_times:
        print('Thread {0} done ({1}).'.format(th_idx, times))
    return times


if __name__ == '__main__':
    pass
